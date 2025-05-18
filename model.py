import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, efficientnet_b0, swin_t
import math

# ------------------------------------------------------------
# Класс ArcMarginProduct реализует ArcFace — модифицированную
# классификационную голову, которая используется для увеличения
# межклассовой разницы между признаками объектов. Это полезно
# для задач, где требуется более точное различие между похожими классами.
# Подробнее: вместо обычной линейной классификации применяется
# косинусная мера с угловым отступом, что делает границы более жёсткими.
# ------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features     # Количество признаков на входе (выход из backbone-сети)
        self.out_features = out_features   # Количество классов, между которыми должна различать модель
        self.s = s                         # Масштабный коэффициент, усиливающий логиты перед softmax
        self.m = m                         # Угловой отступ, который мы вычитаем из угла между вектором признаков и вектором класса

        # Инициализируем обучаемые веса классификатора: один вектор на каждый класс
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # Инициализация по Ксавье — стандартная для линейных слоёв

        self.easy_margin = easy_margin     # Флаг, влияющий на то, как обрабатываются отрицательные значения косинуса

        # Предрасчёт тригонометрических значений, чтобы ускорить инференс
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)    # Порог, ниже которого phi заменяется на другую формулу
        self.mm = math.sin(math.pi - m) * m  # Значение для коррекции phi, когда угол большой

    def forward(self, input, label=None):
        # 1. Нормализуем входные признаки и веса, чтобы получить значения косинуса угла между ними
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))

        # 2. Если мы не обучаем модель (например, при валидации), просто масштабируем косинус и возвращаем его
        if label is None or not self.training:
            return cosine * self.s

        # 3. Вычисляем синус по формуле sin² = 1 - cos²
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))

        # 4. phi — модифицированное значение косинуса с отступом m: cos(θ + m) = cosθ·cosm - sinθ·sinm
        phi = cosine * self.cos_m - sine * self.sin_m

        # 5. Если easy_margin включён — просто заменяем значения на phi, только если cos > 0
        # иначе используем более сложную корректировку для отрицательных значений
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 6. Создаём one-hot представление меток (для каждого примера ставим 1 на позиции правильного класса)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # 7. Подменяем логиты целевого класса на phi, остальные оставляем косинусными
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 8. Возвращаем масштабированные логиты — подаются дальше в softmax
        output *= self.s
        return output


# ------------------------------------------------------------
# Класс SwinTinyExtractor — обёртка для Swin Transformer Tiny
# Удаляет финальную классификационную голову и добавляет слой нормализации.
# Используется, если модель применяется в качестве feature extractor,
# например, для ArcFace.
# ------------------------------------------------------------
class SwinTinyExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.head = nn.Identity()  # Заменяем классификатор на пустую операцию
        self.norm = nn.LayerNorm(768)    # Нормализуем выход: Swin-Tiny выдаёт 768 признаков

    def forward(self, x):
        x = self.model(x)      # Получаем признаки из модели (после Global Pooling)
        x = self.norm(x)       # Применяем LayerNorm, чтобы стабилизировать значения
        return x


# ------------------------------------------------------------
# Универсальная функция get_model загружает одну из трёх моделей:
# ResNet-50, EfficientNet-B0 или Swin Transformer Tiny,
# с возможностью использовать ArcFace голову вместо стандартного классификатора.
# ------------------------------------------------------------
def get_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    dropout_p: float = 0.5,
    freeze_base: bool = False,
    arcface: bool = False
):
    # === ЗАГРУЗКА RESNET-50 ===
    if model_name == "resnet50":
        model = resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features  # Последний линейный слой получает столько входов

        if freeze_base:
            # Замораживаем все веса — нужно для обучения только классификатора
            for param in model.parameters():
                param.requires_grad = False

        if arcface:
            # Если ArcFace, убираем fc, вставляем BatchNorm
            feature_extractor = nn.Sequential(
                *list(model.children())[:-1],  # Убираем avgpool и fc
                nn.Flatten(),
                nn.BatchNorm1d(in_features)    # Стабилизация признаков
            )
        else:
            model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))

    # === ЗАГРУЗКА EFFICIENTNET-B0 ===
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False

        if arcface:
            # Обрезаем классификатор, оставляем фичи + глобальный pooling + norm
            feature_extractor = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.BatchNorm1d(in_features)
            )
        else:
            model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))

    # === ЗАГРУЗКА SWIN-TINY ===
    elif model_name == "swin_tiny":
        model = swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        in_features = model.head.in_features

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False

        if arcface:
            model.head = nn.Identity()  # Убираем стандартную голову
            feature_extractor = SwinTinyExtractor(model)  # Используем нормализующий обёртчик
        else:
            model.head = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))

    else:
        # Ошибка: неверное имя модели
        raise ValueError(f"Неизвестная модель: {model_name}. Доступные варианты: 'resnet50', 'efficientnet_b0', 'swin_tiny'")

    # Возвращаем модель: либо с ArcFace, либо стандартную
    if arcface:
        arc_head = ArcMarginProduct(in_features, num_classes)
        return feature_extractor, arc_head, in_features
    else:
        return model, None, None


# ------------------------------------------------------------
# Основной блок: если запустить файл напрямую — можно протестировать,
# как создаётся и конфигурируется модель, в том числе с ArcFace.
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Тестовая обёртка для проверки модели")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "swin_tiny"], help="Выбор архитектуры")
    parser.add_argument("--num_classes", type=int, default=10, help="Число целевых классов")
    parser.add_argument("--no_pretrained", action='store_true', help="Не использовать предобученные веса")
    parser.add_argument("--dropout", type=float, default=0.5, help="Вероятность дропаут слоя перед классификатором")
    parser.add_argument("--freeze_base", action='store_true', help="Заморозить базовую часть модели")
    parser.add_argument("--arcface", action='store_true', help="Использовать ArcFace голову")
    args = parser.parse_args()

    pretrained_flag = not args.no_pretrained

    result = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=pretrained_flag,
        dropout_p=args.dropout,
        freeze_base=args.freeze_base,
        arcface=args.arcface
    )

    if args.arcface:
        feature_extractor, arc_head, dim = result
        print(f"✅ Загружен feature extractor: {args.model}, выход: {dim} → ArcFace")
    else:
        model = result
        print(f"✅ Загружена обычная модель: {args.model}")
