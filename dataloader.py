### dataloader.py — модуль подготовки данных для обучения моделей классификации изображений минералов

# === Импорт базовых библиотек ===
import os  # Работа с файловой системой: пути, директории, файлы
import random  # Генератор случайных чисел — используется для задания сида
import numpy as np  # Работа с многомерными массивами и математикой
import torch  # Основной фреймворк для работы с нейросетями (PyTorch)

# === Импорт компонентов из torchvision и PyTorch ===
from torchvision import datasets, transforms  # datasets: структура каталогов; transforms: предобработка изображений
from torch.utils.data import DataLoader  # Класс для пакетной загрузки данных с поддержкой параллельной обработки
from sklearn.utils.class_weight import compute_class_weight  # Вычисление весов классов для компенсации дисбаланса

# === Настройка воспроизводимости (фиксация случайности) ===
# Для того чтобы при каждом запуске скрипта результаты были одинаковыми — задаём фиксированный seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():  # Для всех GPU, если они используются
    torch.cuda.manual_seed_all(SEED)

# === Глобальные параметры — определяют общую конфигурацию загрузчика данных ===
DATASET_DIR = "dataset_10classes"  # Папка с подкаталогами train/val/test
BATCH_SIZE = 32  # Размер мини-батча (кол-во изображений за один прогон)
NUM_WORKERS = 2  # Кол-во потоков при загрузке (увеличивает скорость)
IMAGE_SIZE = 256  # Размер изображений, приводимых к квадрату 256x256

# === Определение классов (названия минералов) и их количества в обучающей выборке ===
# Это позволит задать веса классов для корректировки дисбаланса при обучении
CLASS_NAMES = ["Agat", "Berill", "Gematit", "Kal'tsit", "Kassiterit",
               "Khaltsedon", "Korund", "Kvarts", "Magnetit", "Topaz"]
CLASS_COUNTS = [398, 331, 449, 217, 483, 477, 261, 1371, 378, 703]  # Кол-во изображений в каждом классе

# === Вычисление весов классов ===
# Для балансировки функции потерь (например, CrossEntropy или FocalLoss)
CLASS_WEIGHTS = compute_class_weight(
    class_weight='balanced',  # Метод учёта дисбаланса классов
    classes=np.arange(len(CLASS_NAMES)),  # Индексы классов от 0 до 9
    y=[i for i, count in enumerate(CLASS_COUNTS) for _ in range(count)]  # Расширяем каждый класс по количеству
)
CLASS_WEIGHTS_TENSOR = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)  # Переводим в тензор для передачи в модель

# === Аугментации для ResNet-50 ===
# Набор трансформаций, повышающих разнообразие обучающих изображений
resnet_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),  # Случайное обрезание и ресайз
    transforms.RandomHorizontalFlip(p=0.5),  # Отражение по горизонтали (с вероятностью 50%)
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Лёгкая цветовая вариативность
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),  # Размытие применимо к 20% изображений
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.RandomErasing(p=0.2),  # Случайное зачеркивание части изображения
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Стандартизация как в ImageNet
])

# === Аугментации для Swin-Tiny ===
# Адаптируются в зависимости от текущей эпохи обучения
# Чем больше эпох, тем сильнее вариации в изображениях

def get_swin_transforms(epoch):
    if epoch < 30:
        # Начальные эпохи — мягкие аугментации для плавного старта обучения
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.95, 1.0)),  # Случайная обрезка и ресайз (почти полное изображение)
            transforms.RandomHorizontalFlip(p=0.5),  # Отражение по горизонтали с вероятностью 50%
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),  # Незначительные изменения яркости, контраста и насыщенности
            transforms.ToTensor(),  # Перевод изображения в тензор
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация по статистике ImageNet
        ])
    elif epoch < 80:
        # Средние эпохи — умеренные аугментации для повышения обобщающей способности
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),  # Область кадрирования чуть меньше
            transforms.RandomHorizontalFlip(p=0.5),  # Также отражение
            transforms.RandomRotation(degrees=5),  # Лёгкий поворот в пределах ±5 градусов
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Более сильная цветовая вариация
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),  # Иногда применяем размытие (10%)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Поздние эпохи — агрессивные искажения, чтобы не переобучиться на конкретные детали
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),  # Ещё больше вариативности в обрезке
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Поворот до ±10 градусов
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  # Максимальные цветовые изменения
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),  # Размытие применяется чаще (20%)
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1),  # Случайное стирание прямоугольника на изображении (10%)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# === Аугментации для EfficientNet-B0 ===
# Используется RandAugment — современный метод генерации аугментаций
# Resize делает изображение нужного размера без обрезки

efficientnet_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=7),  # 2 случайные операции с интенсивностью 7
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Преобразования для валидационной и тестовой выборок ===
# Без случайных искажений — только Resize + Crop центра изображения
val_test_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # Приведение короткой стороны к 256 пикселям
    transforms.CenterCrop(IMAGE_SIZE),  # Центрирование — обрезка по центру
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
])

# === Сопоставление имени модели и нужной ей трансформации ===
MODEL_TRANSFORMS = {
    "resnet50": resnet_transforms,
    "efficientnet_b0": efficientnet_transforms,
    # swin_tiny определяется отдельно, через функцию get_swin_transforms(epoch)
}

# === Главная функция — загрузка даталоадеров для обучения, валидации и теста ===
def get_dataloaders(model_name,
                    dataset_dir=DATASET_DIR,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    seed=SEED,
                    epoch=0):
    generator = torch.Generator().manual_seed(seed)  # фиксация генератора случайности PyTorch
    pin_mem = torch.cuda.is_available()  # ускоряет работу, если доступна видеокарта

    # Выбор трансформаций под конкретную модель
    if model_name == "swin_tiny":
        train_tf = get_swin_transforms(epoch)
    else:
        train_tf = MODEL_TRANSFORMS.get(model_name, resnet_transforms)

    # Создание обучающего датасета с нужной трансформацией
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "val"), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform=val_test_transforms)

    # === Создание DataLoader'ов для каждой выборки ===
    # Обучающий загрузчик: включает перемешивание и генератор для воспроизводимости
    train_loader = DataLoader(
        train_dataset,  # входной датасет для обучения
        batch_size=batch_size,  # размер батча, напр. 32 изображений за шаг
        shuffle=True,  # перемешивание данных (важно для предотвращения переобучения)
        num_workers=num_workers,  # количество потоков для ускоренной загрузки данных
        pin_memory=pin_mem,  # ускоряет передачу данных на GPU (если есть)
        persistent_workers=(num_workers > 0),  # не останавливать worker'ы между эпохами (экономия времени)
        generator=generator  # генератор случайных чисел для воспроизводимости shuffle
    )

    # Валидационный загрузчик: порядок сохраняется, без перемешивания
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # порядок важен — для стабильного сравнения между эпохами
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0)
    )

    # Тестовый загрузчик: аналогично валидации
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # отключаем перемешивание для воспроизводимой оценки
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0)
    )

    # Возвращаем загрузчики и веса классов (для функции потерь)
    return train_loader, val_loader, test_loader, CLASS_WEIGHTS_TENSOR

# === Проверка при запуске скрипта напрямую ===
# Загружает по очереди данные для всех моделей и выводит размеры батчей
if __name__ == "__main__":
    for model in ["resnet50", "swin_tiny", "efficientnet_b0"]:
        print("Testing dataloader for model:", model)
        train_loader, val_loader, test_loader, weights = get_dataloaders(model_name=model)
        for name, loader in zip(["Train", "Val", "Test"], [train_loader, val_loader, test_loader]):
            images, labels = next(iter(loader))  # Получение одного батча
            print(f"{name} batch: images={images.shape}, labels={labels.shape}")
        print("Class weights:", weights.tolist())
