import logging  # Библиотека для логирования (записи действий программы в файл или консоль)
from train import train_model  # Импорт функции обучения модели из модуля train.py
from evaluate import evaluate_model, save_f1_summary  # Импорт функций оценки модели и сохранения итогов

# === Функция для настройки логирования ===
def setup_logging():
    # Настройка логирования: вывод логов в файл и на экран
    logging.basicConfig(
        level=logging.INFO,  # Уровень логирования: INFO — средний уровень подробности
        format="%(asctime)s [%(levelname)s] %(message)s",  # Формат записи логов (время, уровень, сообщение)
        datefmt="%Y-%m-%d %H:%M:%S",  # Формат отображения даты и времени
        handlers=[
            logging.FileHandler("results/eval.log", mode='w', encoding='utf-8'),  # Файл для логов
            logging.StreamHandler()  # Вывод в консоль
        ]
    )

# === Основная функция, запускаемая при старте скрипта ===
def main():
    setup_logging()  # Запускаем логирование

    # Список моделей, которые будем обучать и оценивать
    models = ["resnet50", "efficientnet_b0", "swin_tiny"]
    f1_scores = {}  # Словарь для хранения итоговых F1-оценок по каждой модели

    logging.info("=== ШАГ 1: Обучение моделей с ArcFace ===")
    for model_name in models:
        train_model(model_name=model_name)  # Запускаем обучение модели

    logging.info("=== ШАГ 2: Оценка обученных моделей с ArcFace ===")
    for model_name in models:
        logging.info(f"🔍 Оценка модели: {model_name}")
        avg_f1 = evaluate_model(model_name=model_name)  # Получаем среднюю F1-оценку
        f1_scores[model_name] = avg_f1  # Сохраняем её в словарь

    logging.info("=== ШАГ 3: Сравнение F1-метрик ===")
    save_f1_summary(f1_scores)  # Сохраняем сравнение в таблицу и строим график

    logging.info("\u2705 Обучение и оценка завершены успешно")

# === Стартовая точка программы ===
if __name__ == "__main__":
    main()  # Запускаем основную функцию
