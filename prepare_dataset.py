# Подключаем необходимые библиотеки
import os  # работа с файловой системой
import shutil  # копирование и удаление файлов/папок
import pandas as pd  # работа с таблицами (DataFrame)
import numpy as np  # работа с массивами чисел
from PIL import Image  # работа с изображениями (Pillow)
import cv2  # библиотека OpenCV для обработки изображений
import ast  # безопасный парсинг строк в Python-объекты
from sklearn.model_selection import train_test_split  # разбиение на train/val/test
from unidecode import unidecode  # перевод кириллицы в латиницу
from collections import defaultdict  # словарь с значениями по умолчанию
from datetime import datetime  # для логирования по времени
from tqdm import tqdm  # прогресс-бары в циклах

# === НАСТРОЙКИ ===
# Пути к CSV-файлам с данными
MINERALS_10_PATH = "minerals_10.csv"
MINERALS_FULL_PATH = "minerals_full.csv"
# Папка, где хранятся исходные изображения минералов
ROOT_IMAGE_DIR = "C:/Users/KROL/PycharmProjects/CLASS10/Minerals_2023/mineral_images"
# Куда будут сохраняться подготовленные изображения
SPLIT_OUTPUT_DIR = "dataset_10classes"
# Пропорции разбиения на обучающую, валидационную и тестовую выборки
SPLIT_RATIOS = (0.7, 0.2, 0.1)
RANDOM_STATE = 42  # фиксируем рандом для воспроизводимости

# Параметры обработки изображений
MIN_SIZE = 256  # минимальный размер по ширине/высоте
MARGIN_PERCENT = 0.1  # процент на расширение области объекта (bbox)
CONTRAST_STD_THRESHOLD = 60.0  # порог для проверки контраста изображения
BLUR_THRESHOLD = 50.0  # порог для проверки размытости
MIN_AREA_RATIO = 0.25  # минимальная площадь объекта в кадре
MIN_SIDE_RATIO = 0.35  # минимальное соотношение сторон bbox

# === ЛОГИ ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # создаем директорию логов
LOG_PATH = os.path.join(LOG_DIR, "dataset_preparation.log")  # лог-файл событий
CSV_PATH = os.path.join(LOG_DIR, "removed_images.csv")  # файл с удаленными изображениями
SUMMARY_PATH = os.path.join(LOG_DIR, "summary.txt")  # итоговая сводка

# Логгируем все события в список
log_lines = []
removed_records = []  # сюда записываются данные об удаленных изображениях
summary_stats = defaultdict(lambda: {'total': 0, 'saved': 0, 'removed': 0})  # статистика по выборке
reason_stats = defaultdict(int)  # статистика по причинам удаления

# Удобная функция для записи лога с временной меткой
def log(msg):
    log_lines.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")

# Фиксируем удаление изображения с объяснением причины
def record_removal(path, split, cls, reason):
    removed_records.append({
        'path': path, 'split': split, 'class': cls, 'reason': reason
    })
    reason_stats[reason] += 1

# === ФУНКЦИИ ===
# Функция для вычисления доминирующего цвета по краям изображения (используется как фон при дополнении до минимального размера)
def get_background_color(pil_img):
    img = np.array(pil_img)
    h, w = img.shape[:2]
    top = img[:20, :, :] if h >= 20 else img[:h//2, :, :]
    bottom = img[-20:, :, :] if h >= 20 else img[h//2:, :, :]
    left = img[:, :20, :] if w >= 20 else img[:, :w//2, :]
    right = img[:, -20:, :] if w >= 20 else img[:, w//2:, :]
    edge_px = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3),
                              left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    return tuple(np.median(edge_px, axis=0).astype(np.uint8).tolist())

# Дополнение изображения до минимального размера MIN_SIZE с фоном, подобранным по краю
def ensure_min_size(pil_img):
    w, h = pil_img.size
    if w >= MIN_SIZE and h >= MIN_SIZE:
        return pil_img
    new_w = max(w, MIN_SIZE)
    new_h = max(h, MIN_SIZE)
    bg_color = get_background_color(pil_img)
    canvas = Image.new("RGB", (new_w, new_h), bg_color)
    canvas.paste(pil_img, ((new_w - w) // 2, (new_h - h) // 2))
    return canvas

# Повышение контрастности с помощью CLAHE, если изначально изображение слишком блеклое
def apply_clahe_if_needed(pil_img):
    img_np = np.array(pil_img)
    if img_np.size == 0:
        return pil_img
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if np.std(gray) >= CONTRAST_STD_THRESHOLD:
        return pil_img  # изображение уже достаточно контрастное
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))

# Проверка размытости изображения по дисперсии лапласиана (метод OpenCV)
def is_blurry(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

# Увеличиваем область bbox на заданный процент
# Это помогает захватить контекст или слегка выйти за границы объекта
# Также ограничиваем область размерами изображения

def expand_box(box, img_w, img_h):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    dx, dy = int(w * MARGIN_PERCENT), int(h * MARGIN_PERCENT)
    return (
        max(0, x0 - dx),
        max(0, y0 - dy),
        min(img_w, x1 + dx),
        min(img_h, y1 + dy)
    )

# Упорядочиваем координаты bbox, чтобы всегда было (x0,y0,x1,y1), даже если передано наоборот
def normalize_box(box):
    x0, y0, x1, y1 = box
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

# Функция для вычисления IOU (перекрытия) двух прямоугольников — нужна для объединения близких боксов
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Объединяем два пересекающихся бокса в один, берём максимальную уверенность и объединяем координаты
def merge_boxes(b1, b2):
    return {
        "label": b1.get("label", "") or b2.get("label", ""),
        "confidence": max(b1.get("confidence", 0), b2.get("confidence", 0)),
        "box": normalize_box([
            min(b1["box"][0], b2["box"][0]),
            min(b1["box"][1], b2["box"][1]),
            max(b1["box"][2], b2["box"][2]),
            max(b1["box"][3], b2["box"][3]),
        ])
    }

# Убираем дубликаты bbox с высоким IOU и объединяем их
def deduplicate_boxes(boxes, iou_threshold=0.5):
    merged = []
    used = set()
    for i, b1 in enumerate(boxes):
        if i in used:
            continue
        box1 = normalize_box(b1["box"])
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            box2 = normalize_box(boxes[j]["box"])
            if iou(box1, box2) > iou_threshold:
                b1 = merge_boxes(b1, boxes[j])
                used.add(j)
                box1 = normalize_box(b1["box"])
        merged.append(b1)
    return merged

# === УДАЛЕНИЕ СТАРЫХ ДАННЫХ ===
# Если уже существует директория с готовым датасетом, удаляем её, чтобы не накапливать старые данные
if os.path.exists(SPLIT_OUTPUT_DIR):
    log(f"[🧹] Удаляем старую директорию: {SPLIT_OUTPUT_DIR}")
    shutil.rmtree(SPLIT_OUTPUT_DIR)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)  # Создаём папку заново

# === ЗАГРУЗКА CSV ===
# Загружаем 2 таблицы: первая — отфильтрованные id (10 классов), вторая — полные аннотации
# Затем объединяем их по id

df_10 = pd.read_csv(MINERALS_10_PATH)  # отобранные id (по 10 классам)
df_full = pd.read_csv(MINERALS_FULL_PATH)  # все изображения и аннотации

# Объединяем два датафрейма по id, сохраняем путь к изображению и метку
# path_y — путь к изображению, ru_name — название минерала (на русском)
df = df_10.merge(df_full, on="id")
df = df[["id", "path_y", "ru_name"]].rename(columns={"path_y": "image_path", "ru_name": "label"})

# Преобразуем русские названия в латиницу (для удобства папок)
df["label_en"] = df["label"].apply(lambda x: unidecode(str(x).strip()).replace(" ", "_"))

# === РАЗБИВАЕМ НА TRAIN / VAL / TEST ===
# Используем стратифицированное разбиение по сбалансированным классам
train_df, temp_df = train_test_split(
    df,
    stratify=df['label_en'],
    test_size=1 - SPLIT_RATIOS[0],
    random_state=RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df,
    stratify=temp_df['label_en'],
    test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
    random_state=RANDOM_STATE
)
splits = {"train": train_df, "val": val_df, "test": test_df}  # сохраняем в словарь

# === КОПИРОВАНИЕ И ОБРАБОТКА ===
# Проходим по каждой выборке (train/val/test), создаем соответствующие папки, копируем изображения,
# проверяем корректность bbox, качество изображения и вырезаем области интереса

for split_name, split_df in splits.items():
    log(f"--- Обработка выборки: {split_name}, {len(split_df)} изображений ---")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"{split_name}", ncols=80):

        label = row["label_en"]
        src = os.path.join(ROOT_IMAGE_DIR, row["image_path"])  # путь к оригинальному изображению
        dst_dir = os.path.join(SPLIT_OUTPUT_DIR, split_name, label)  # директория, куда будет сохранено изображение
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(row["image_path"]))  # путь назначения файла

        summary_stats[(split_name, label)]['total'] += 1  # увеличиваем счётчик обработанных изображений

        try:
            shutil.copy2(src, dst)  # копируем изображение
        except Exception as e:
            log(f"[ERR] Не удалось скопировать {src} → {e}")
            record_removal(src, split_name, label, "ошибка копирования")
            continue

        # Ищем запись в полной таблице по имени файла
        row_full = df_full[df_full["path"].str.endswith(os.path.basename(row["image_path"]))].head(1)
        if row_full.empty:
            log(f"[DEL] Нет строки в CSV: {dst}")
            os.remove(dst)
            record_removal(dst, split_name, label, "нет строки в CSV")
            continue

        row_data = row_full.iloc[0]  # берём найденную строку

        # Если нет аннотаций — удаляем изображение
        if pd.isna(row_data.get("mineral_boxes", None)):
            log(f"[DEL] Нет box'ов: {dst}")
            os.remove(dst)
            record_removal(dst, split_name, label, "нет box'ов")
            continue

        try:
            boxes = deduplicate_boxes(ast.literal_eval(row_data["mineral_boxes"]))
        except:
            log(f"[DEL] Ошибка разбора box'ов: {dst}")
            os.remove(dst)
            record_removal(dst, split_name, label, "ошибка в box'ах")
            continue

        try:
            img = Image.open(dst).convert("RGB")
        except:
            log(f"[DEL] Не удалось открыть изображение: {dst}")
            os.remove(dst)
            record_removal(dst, split_name, label, "не читается изображение")
            continue

        W, H = img.size  # размеры изображения
        base_name = os.path.splitext(os.path.basename(dst))[0]  # имя файла без расширения
        saved = False  # флаг, сохранили ли хоть один crop

        for idx, b in enumerate(boxes):
            if "box" not in b or len(b["box"]) != 4:
                continue

            # преобразуем координаты из нормированных в абсолютные значения пикселей
            x0 = int(b["box"][0] * W)
            y0 = int(b["box"][1] * H)
            x1 = int(b["box"][2] * W)
            y1 = int(b["box"][3] * H)
            w, h = x1 - x0, y1 - y0
            area = (w * h) / (W * H)

            # фильтруем слишком маленькие объекты (по площади и по сторонам)
            if area < MIN_AREA_RATIO and w < MIN_SIDE_RATIO * W and h < MIN_SIDE_RATIO * H:
                continue

            # Расширяем область и вырезаем изображение
            x0, y0, x1, y1 = expand_box((x0, y0, x1, y1), W, H)
            crop = img.crop((x0, y0, x1, y1))
            crop = apply_clahe_if_needed(crop)  # контрастирование

            if is_blurry(crop):  # фильтрация по размытости
                continue

            crop = ensure_min_size(crop)  # дополнение до нужного размера
            out_path = os.path.join(dst_dir, f"{base_name}_crop{idx}.jpg")  # имя файла
            crop.save(out_path)  # сохраняем финальный crop
            log(f"[OK] Сохранено: {out_path}")
            summary_stats[(split_name, label)]['saved'] += 1
            saved = True

            os.remove(dst)  # удаляем оригинал (мы сохранили только нужные вырезки)
            log(f"[DEL] Удалено оригинальное изображение: {dst}")
            if not saved:
                record_removal(dst, split_name, label, "ни один crop не сохранён")

    # === СОХРАНЕНИЕ ЛОГОВ ===
    # Сохраняем текстовый лог всех событий
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Сохраняем таблицу с удалёнными изображениями и причинами удаления
    pd.DataFrame(removed_records).to_csv(CSV_PATH, index=False, encoding="utf-8")

    # Сохраняем сводную текстовую статистику по сохранённым изображениям
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        # === Раздел 1: Общая статистика по выборкам и классам
        total_all = 0
        split_class_counts = defaultdict(lambda: defaultdict(int))

        for (split, cls), stat in sorted(summary_stats.items()):
            split_class_counts[split][cls] += stat['saved']
            total_all += stat['saved']

        for split in ["train", "val", "test"]:
            split_name = split.upper()
            split_total = sum(split_class_counts[split].values())
            f.write(f"📁 {split_name} — {split_total} images\n")
            for cls in sorted(split_class_counts[split]):
                f.write(f"  {cls}: {split_class_counts[split][cls]}\n")
            f.write("\n")

        f.write(f"📊 TOTAL: {total_all} images in all splits\n\n")

        # === Раздел 2: Статистика по сохранению/удалению
        f.write("📊 СТАТИСТИКА ПОТЕРЬ И СОХРАНЕНИЯ:\n\n")
        f.write(f"{'| Split':<8}| {'Class':<12}| {'Total':>5} | {'Saved':>5} | {'Lost':>4} | {'Saved %':>7} |\n")
        f.write("|" + "-" * 7 + "|" + "-" * 13 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 6 + "|" + "-" * 9 + "|\n")
        for (split, cls), stat in sorted(summary_stats.items()):
            total = stat['total']
            saved = stat['saved']
            lost = total - saved
            perc = 100 * saved / total if total else 0
            f.write(f"| {split:<6}| {cls:<12}| {total:>5} | {saved:>5} | {lost:>4} | {perc:>6.1f}% |\n")
        f.write("\n")

        # === Раздел 3: Подсчёт причин удаления
        f.write("🗑️ ПРИЧИНЫ УДАЛЕНИЯ:\n")
        for reason, count in sorted(reason_stats.items(), key=lambda x: -x[1]):
            f.write(f"{reason}: {count} файлов\n")

    # Финальное сообщение в лог
    doc_log = "[✔] Обработка завершена. Логи и отчёты сохранены."
    log(doc_log)
    print(doc_log)



