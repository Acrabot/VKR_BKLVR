# train.py — основной скрипт обучения моделей

# === Импорт базовых модулей ===
import os                   # Работа с файловой системой
import logging              # Для логирования информации (например, в консоль или файл)
from datetime import datetime  # Для записи текущей даты и времени

# === Импорт PyTorch и инструментов обучения ===
import torch
import torch.nn as nn                       # Нейросетевые слои и функции
import torch.optim as optim                 # Оптимизаторы, например AdamW
from torch.utils.tensorboard import SummaryWriter  # Для визуализации через TensorBoard
from tqdm import tqdm                       # Удобная прогресс-бар библиотека
from torch.amp import autocast, GradScaler  # Для ускорения обучения через AMP (автоматическую смешанную точность)
from sklearn.metrics import f1_score        # Метрика качества классификации

# === Подавление предупреждений от PyTorch ===
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# === Импорт кастомных модулей проекта ===
from dataloader import get_dataloaders       # Функция, возвращающая загрузчики данных
from model import get_model                  # Функция, создающая нейросетевую модель

# === Реализация FocalLoss — улучшенная версия CrossEntropy для дисбалансированных классов ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha       # Веса классов (для компенсации дисбаланса)
        self.gamma = gamma       # Параметр фокусировки (больше значение -> сильнее фокус на сложных примерах)
        self.reduction = reduction  # Метод усреднения: 'mean', 'sum', либо ничего

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # вероятность правильного класса
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# === Функция заморозки BatchNorm слоёв (используется при transfer learning) ===
def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            for param in module.parameters():
                param.requires_grad = False  # Отключаем обучение для параметров BatchNorm

# === Функция разморозки BatchNorm слоёв ===
def unfreeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.train()  # Включаем режим обучения
            for param in module.parameters():
                param.requires_grad = True

# === Функция для группировки слоёв по уровням (для LLRD — layer-wise learning rate decay) ===
def get_layer_groups_for_llrd(features, model_name):
    if model_name == "efficientnet_b0":
        if isinstance(features, nn.Sequential):
            return list(features[0].children())
        else:
            return list(features.features.children())  # для EfficientNet без ArcFace
    elif model_name == "swin_tiny":
        return list(features.model.features.children())  # специфичный доступ к Swin-T
    else:
        return list(features.children())  # стандартный случай (например, ResNet)


# === Главная функция обучения модели ===
def train_model(model_name):
    # Настройки путей и параметров обучения
    dataset_dir = "dataset_10classes"                # Папка с подготовленным датасетом
    model_dir = "models"                             # Папка для сохранения моделей (чекпоинтов)
    batch_size = 32                                   # Размер батча (сколько изображений одновременно подаётся в модель)

    # Устанавливаем параметры обучения для каждой модели индивидуально
    if model_name == "resnet50":
        epochs = 80                                    # Общее количество эпох
        patience = 15                                  # Количество эпох без улучшения, после которых срабатывает early stopping
        warmup_epochs = 5                              # Сколько эпох модель "разогревается" перед включением ArcFace
        base_lr = 1.5e-4                               # Базовая скорость обучения
        llrd_decay = 0.8                               # Фактор затухания скорости обучения по слоям
        freeze_schedule = [5]                          # На какой эпохе размораживать слои
    elif model_name == "efficientnet_b0":
        epochs = 90
        patience = 20
        warmup_epochs = 5
        base_lr = 2e-4
        llrd_decay = 0.9
        freeze_schedule = [5]
    elif model_name == "swin_tiny":
        epochs = 150
        patience = 35
        warmup_epochs = 10
        base_lr = 2e-4
        llrd_decay = 0.95
        freeze_schedule = []  # swin обучается сразу полностью

    # Прочие общие настройки
    weight_decay = 1e-4                                # L2-регуляризация
    num_workers = 2                                    # Количество потоков при загрузке данных
    use_amp = True                                     # Использовать автоматическую смешанную точность
    use_focal = True                                   # Использовать Focal Loss (иначе CrossEntropyLoss)

    os.makedirs(model_dir, exist_ok=True)              # Создаём папку, если не существует
    now = datetime.now().strftime("%Y%m%d_%H%M%S")     # Время запуска (для логов TensorBoard)
    writer = SummaryWriter(log_dir=os.path.join("runs", f"{now}_{model_name}_arcface"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Используем GPU при наличии
    logging.info(f"🚀 Начинаем обучение: {model_name} on {device}")

    # Загружаем данные и веса классов (для взвешенной функции потерь)
    train_loader, val_loader, _, class_weights = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
        epoch=0  # для Swin выбираются аугментации по эпохам
    )
    logging.info("Class weights: [%s]", ", ".join(f"{w:.3f}" for w in class_weights.tolist()))

    freeze_base = model_name != "swin_tiny"           # только swin не требует заморозки на старте
    dropout_value = 0.2 if model_name == "swin_tiny" else 0.5  # индивидуальное значение dropout

    # Получаем архитектуру модели (features — экстрактор, arc_head — ArcFace классификатор, feat_dim — размер выхода)
    if model_name == "swin_tiny":
        features, arc_head, feat_dim = get_model(model_name, freeze_base=False, arcface=True, dropout_p=dropout_value)
        logging.info("✅ Swin Transformer сразу обучается с ArcFace.")
    else:
        features, arc_head, feat_dim = get_model(model_name, freeze_base=freeze_base, arcface=False, dropout_p=dropout_value)
        logging.info("🧪 Warm-up начат: ArcFace отключён, используется стандартная голова классификации.")

    # Переносим модель и голову на нужное устройство
    features.to(device)
    if arc_head is not None:
        arc_head.to(device)

    # Замораживаем BatchNorm если нужно
    if freeze_base:
        freeze_batchnorm(features)
    else:
        unfreeze_batchnorm(features)

    # Выбор функции потерь: Focal или CrossEntropy
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=1.5) if use_focal else nn.CrossEntropyLoss(weight=class_weights.to(device))
    logging.info("🎯 Используется FocalLoss с gamma=1.5" if use_focal else "📘 Используется стандартная CrossEntropyLoss")

    # === Настройка оптимизатора с LLRD ===
    # LLRD (Layer-wise Learning Rate Decay) — постепенное уменьшение скорости обучения для нижних слоёв
    layer_groups = get_layer_groups_for_llrd(features, model_name)  # разбиваем модель на блоки
    param_groups = []  # список параметров с индивидуальной скоростью обучения
    for i, layer in enumerate(layer_groups):
        decay_factor = llrd_decay ** (len(layer_groups) - i - 1)  # больше глубина — меньше lr
        lr = base_lr * decay_factor
        param_groups.append({"params": layer.parameters(), "lr": lr})

    if arc_head is not None:
        param_groups.append(
            {"params": arc_head.parameters(), "lr": base_lr})  # классификатор всегда обучается с base_lr

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)  # Оптимизатор AdamW

    # === Планировщик скорости обучения с warm-up и cosine annealing ===
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)  # сначала lr плавно растёт
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)  # потом затухает по косинусной кривой
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs], last_epoch=-1)  # комбинируем

    skip_scheduler_step = False  # если True, то пропускаем один шаг обновления планировщика (например, после перезапуска)
    scaler = GradScaler(enabled=use_amp)  # скейлер для AMP
    best_val_f1 = 0.0  # Лучшее значение F1 на валидации
    best_val_loss = float('inf')  # Лучшее значение loss на валидации
    epochs_no_improve = 0  # Счётчик эпох без улучшений

    # === Основной цикл по эпохам ===
    for epoch in range(1, epochs + 1):
        skipped_steps = 0  # Количество шагов, которые были пропущены из-за отсутствия градиентов
        logging.info(f"▶️ Epoch {epoch}/{epochs} ({round(100 * epoch / epochs)}%) — {model_name}")

        # === Переход от warm-up к полной разморозке и ArcFace ===
        if freeze_base and epoch in freeze_schedule:
            for param in features.parameters():
                param.requires_grad = True  # Разморозка слоёв
            unfreeze_batchnorm(features)  # Также активируем BatchNorm
            freeze_base = False
            logging.info(f"🔓 Полная разморозка {model_name} на эпохе {epoch}")
            logging.info("🔁 Завершение warm-up: включаем ArcFace и полную классификационную голову.")

            # Повторно загружаем модель уже с ArcFace-головой
            features, arc_head, feat_dim = get_model(model_name, freeze_base=False, arcface=True)
            features.to(device)
            arc_head.to(device)

            # Повторно создаём optimizer и scheduler, так как параметры изменились
            layer_groups = get_layer_groups_for_llrd(features, model_name)
            param_groups = []
            for i, layer in enumerate(layer_groups):
                decay_factor = llrd_decay ** (len(layer_groups) - i - 1)
                lr = base_lr * decay_factor
                param_groups.append({"params": layer.parameters(), "lr": lr})
            if arc_head is not None:
                param_groups.append({"params": arc_head.parameters(), "lr": base_lr})

            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                     milestones=[warmup_epochs], last_epoch=epoch - 1)
            skip_scheduler_step = True  # пропустить обновление lr на этой итерации
            scaler = GradScaler(enabled=use_amp)
            logging.info("✅ ArcFace успешно включён и optimizer/scheduler пересозданы.")

        # === Режим обучения для всех компонентов ===
        features.train()
        if arc_head is not None:
            arc_head.train()

        running_loss = 0  # Суммарный loss за эпоху (будет усреднён)
        correct = 0  # Количество правильно предсказанных примеров
        total = 0  # Общее количество примеров
        all_preds = []  # Список предсказанных меток
        all_labels = []  # Список истинных меток

        epoch_bar = tqdm(train_loader, desc="🔠", dynamic_ncols=True)  # Прогресс-бар
        for imgs, lbls in epoch_bar:
            imgs, lbls = imgs.to(device), lbls.to(device)  # Перенос на GPU
            optimizer.zero_grad()  # Обнуляем градиенты

            # === Включаем AMP (автоматическую смешанную точность) ===
            with autocast(device_type='cuda', enabled=use_amp):
                feats = features(imgs)  # Получаем выход feature extractor'а
                if feats.ndim > 2:
                    feats = feats.view(feats.size(0), -1)  # Преобразуем тензор в [batch, features] при необходимости
                if arc_head is not None:
                    outputs = arc_head(feats, lbls)  # ArcFace классификация
                else:
                    outputs = feats  # Стандартная голова (линейный слой в features)

                loss = criterion(outputs, lbls)  # Вычисляем функцию потерь

            scaler.scale(loss).backward()  # backward() с учетом AMP

            if use_amp:
                # Проверка: есть ли градиенты (некоторые параметры могли быть заморожены)
                has_grad = any(
                    p.grad is not None
                    for group in optimizer.param_groups
                    for p in group['params']
                    if p.requires_grad
                )
                if has_grad:
                    scaler.step(optimizer)  # шаг оптимизатора
                    scaler.update()
                else:
                    skipped_steps += 1  # если градиентов не было, пропускаем шаг
            else:
                optimizer.step()  # обычный шаг (без AMP)

            # === Планировщик lr ===
            if not skip_scheduler_step:
                scheduler.step()
            else:
                skip_scheduler_step = False  # сбрасываем флаг, чтобы scheduler продолжил работу на следующих итерациях

            # === Логирование статистики ===
            running_loss += loss.item() * imgs.size(0)  # Умножаем на размер батча — получаем абсолютный loss
            with torch.no_grad():
                preds = outputs.detach().argmax(dim=1)  # Предсказанные классы (индекс максимального логита)
                correct += (preds == lbls).sum().item()  # Подсчёт верных
                total += lbls.size(0)  # Общее количество
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(lbls.cpu().tolist())
# === Подсчёт метрик за эпоху ===
        epoch_loss = running_loss / total                        # Среднее значение функции потерь по всем обучающим примерам
        epoch_acc = correct / total                              # Доля правильно классифицированных примеров на обучающей выборке
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')  # Усреднённый по классам F1-скор (макро-F1)

        # Вывод в лог сводной информации об обучении
        logging.info(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}")

        # Сохраняем метрики для TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch)  # График loss по эпохам
        writer.add_scalar("Acc/train", epoch_acc, epoch)    # График точности по эпохам
        writer.add_scalar("F1/train", epoch_f1, epoch)      # График F1 по эпохам

        # === Оценка модели на валидационной выборке ===
        features.eval()                          # Переводим feature extractor в режим оценки (обнуляет dropout, фиксирует batchnorm)
        if arc_head is not None:
            arc_head.eval()                      # Аналогично переводим ArcFace-классификатор

        val_loss = 0                             # Суммарная функция потерь на валидации
        correct = 0                              # Счётчик правильно классифицированных примеров
        total = 0                                # Общее количество примеров
        val_preds = []                           # Предсказанные метки классов
        val_labels = []                          # Истинные метки классов

        # Отключаем подсчёт градиентов для ускорения и экономии памяти
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)  # Загружаем изображения и метки на устройство
                feats = features(imgs)                          # Прогоняем изображения через feature extractor
                if feats.ndim > 2:                              # Если выход многомерный (например, B×C×H×W), то выравниваем
                    feats = feats.view(feats.size(0), -1)

                if arc_head is not None:
                    outputs = arc_head(feats, lbls)            # Классификация через ArcFace (если уже включена)
                else:
                    outputs = feats                            # Иначе берём прямой выход feature extractor’а

                loss = criterion(outputs, lbls)                # Вычисляем функцию потерь
                val_loss += loss.item() * imgs.size(0)         # Учитываем loss с весом по размеру батча

                preds = outputs.detach().argmax(dim=1)         # Находим предсказанные классы (индекс максимального логита)
                correct += (preds == lbls).sum().item()        # Увеличиваем счётчик правильных предсказаний
                total += lbls.size(0)                           # Обновляем общее количество образцов
                val_preds.extend(preds.cpu().tolist())         # Сохраняем все предсказания (для метрик)
                val_labels.extend(lbls.cpu().tolist())         # Сохраняем все метки

        # Подсчёт итоговых валидационных метрик
        val_loss /= total                                      # Средний loss на валидации
        val_acc = correct / total                              # Доля правильных
        val_f1 = f1_score(val_labels, val_preds, average='macro')  # Макро-F1 на валидации

        # Вывод результатов валидации в лог и TensorBoard
        logging.info(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[-1]['lr'], epoch)  # Логируем текущую скорость обучения

        # === Условие улучшения — сохраняем модель только при росте F1
        if val_f1 > best_val_f1 + 1e-4:  # Порог 1e-4 защищает от сохранения из-за случайных флуктуаций
            best_val_f1 = val_f1
            epochs_no_improve = 0  # Сброс счётчика отсутствия прогресса

            # Собираем словарь с состоянием модели, оптимизатора и прочего
            ckpt = {
                'epoch': epoch,
                'features_state': features.state_dict(),       # Веса feature extractor'а
                'optim_state': optimizer.state_dict(),         # Состояние оптимизатора
                'sched_state': scheduler.state_dict(),         # Состояние планировщика
                'scaler_state': scaler.state_dict(),           # Состояние AMP scaler’а
                'best_val_f1': best_val_f1,                    # Наилучшее значение F1
                'best_val_loss': best_val_loss                 # (опционально, может быть использовано потом)
            }
            if arc_head is not None:
                ckpt['arc_head_state'] = arc_head.state_dict()  # Сохраняем ArcFace голову, если есть

            # Сохраняем чекпоинт с указанием эпохи и F1 в названии файла
            archive_path = os.path.join(model_dir, f"{model_name}_arcface_ep{epoch}_f1{val_f1:.4f}.pth")
            torch.save(ckpt, archive_path)
            logging.info(f"📦 Checkpoint saved (F1 improved): {archive_path}")

            # Также сохраняем файл без привязки к эпохе — для быстрой загрузки последней лучшей модели
            simple_path = os.path.join(model_dir, f"{model_name}_arcface.pth")
            torch.save(ckpt, simple_path)
            logging.info(f"📄 Simple checkpoint saved: {simple_path}")

        else:
            epochs_no_improve += 1  # Если F1 не улучшилось, увеличиваем счётчик стагнации
            logging.info(f"No Val F1 improvement: {epochs_no_improve}/{patience}")

        # === Early Stopping — прекращаем обучение, если модель не улучшается N эпох подряд
        if epochs_no_improve >= patience:
            logging.info(f"🚩 Early stopping на эпохе {epoch}: {patience} эпох без улучшений по Val F1.")
            break  # Прерываем цикл обучения

    # === После завершения всех эпох или Early Stopping ===
    if skipped_steps > 0:
        logging.warning(
            f"⚠️ В эпохе {epoch} пропущено {skipped_steps} scaler.step() — отсутствовали градиенты (warm-up или слои были заморожены).")

    writer.close()  # Завершаем запись TensorBoard
    logging.info(f"✅ Завершено обучение модели: {model_name} с ArcFace")
