# === Импорты и настройки ===
import os  # Модуль для работы с файловой системой: проверка/создание папок, манипуляции с путями
import torch  # Основная библиотека PyTorch для создания и запуска нейросетей
import pandas as pd  # Библиотека для работы с таблицами и сохранения отчётов в .csv/.xlsx
import matplotlib.pyplot as plt  # Библиотека для визуализации графиков и изображений
import seaborn as sns  # Расширение matplotlib: стильные визуализации, особенно для тепловых карт
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, \
    top_k_accuracy_score  # Метрики оценки классификации
from sklearn.manifold import TSNE  # Метод t-SNE для понижения размерности эмбеддингов (визуализация)
import numpy as np  # Библиотека для научных и численных вычислений, массивы и матрицы

# Импорт пользовательских модулей из проекта
from model import get_model  # Функция для загрузки и конфигурации модели по имени
from dataloader import get_dataloaders  # Функция для загрузки обучающей, валидационной и тестовой выборки

# Создаётся директория для хранения всех выходных результатов (если ещё не создана)
os.makedirs("results", exist_ok=True)


# === Функция аугментации для TTA (Test-Time Augmentation) ===
def apply_tta(imgs):
    from torchvision.transforms.functional import rotate  # Импортируем функцию поворота из torchvision

    # Создаём список с оригинальными изображениями
    tta_imgs = [imgs]

    # Добавляем горизонтально отзеркаленные изображения (разворачиваем по последней оси: ширина)
    tta_imgs.append(torch.flip(imgs, dims=[3]))  # Например, превращает левую часть изображения в правую и наоборот

    # Добавляем изображения, повернутые на 90° и 180° по часовой стрелке
    tta_imgs.append(rotate(imgs, angle=90))
    tta_imgs.append(rotate(imgs, angle=180))

    # Возвращаем список с четырьмя версиями одного и того же батча
    return tta_imgs


# === Основная функция оценки модели на тестовых данных ===
def evaluate(features, arc_head, dataloader, device, class_names):
    features.eval()  # Переводим feature-экстрактор в режим оценки (отключаются Dropout, BatchNorm работает иначе)
    arc_head.eval()  # Переводим ArcFace голову в режим оценки
    all_preds = []  # Здесь будут финальные предсказанные классы
    all_probs = []  # Здесь будут вероятности по всем классам для каждого изображения
    all_targets = []  # Здесь будут настоящие метки (ground truth)
    embeddings = []  # Список эмбеддингов изображений, полученных после feature extractor

    with torch.no_grad():  # Отключаем градиенты для повышения производительности и экономии памяти
        for imgs, lbls in dataloader:  # Проходим по всем батчам из тестового даталоадера
            imgs = imgs.to(device)  # Переносим изображения на устройство (GPU или CPU)
            tta_imgs = apply_tta(imgs)  # Генерируем TTA версии изображений
            probs_sum = 0  # Инициализируем сумму вероятностей для усреднения по TTA
            feats_all = []  # Список всех эмбеддингов (на каждую TTA-преобразованную версию)

            for tta in tta_imgs:  # Проходим по каждой TTA-версии изображения
                feats = features(tta).view(tta.size(0), -1)  # Получаем эмбеддинги (вектор признаков) и выпрямляем
                outputs = arc_head(feats)  # Пропускаем через классификатор ArcFace (возвращает логиты)
                probs = torch.softmax(outputs, dim=1)  # Переводим логиты в вероятности через Softmax
                probs_sum += probs  # Накопление вероятностей по TTA, чтобы потом усреднить
                feats_all.append(feats)  # Сохраняем эмбеддинги этой версии

            # Усредняем вероятности по всем TTA
            avg_probs = probs_sum / len(tta_imgs)
            # Усредняем эмбеддинги по всем TTA-версиям
            avg_feats = torch.stack(feats_all).mean(dim=0)

            # Получаем индекс класса с максимальной вероятностью
            preds = avg_probs.argmax(dim=1).cpu()

            # Добавляем предсказания, вероятности и настоящие метки к общему списку
            all_preds.extend(preds.tolist())
            all_probs.append(avg_probs.cpu())
            all_targets.extend(lbls.tolist())
            embeddings.append(avg_feats.cpu())

    # Объединяем батчи в один общий тензор/массив
    probs_tensor = torch.cat(all_probs, dim=0)
    emb_tensor = torch.cat(embeddings, dim=0)

    # Получаем подробный отчёт: precision, recall, f1-score и accuracy по каждому классу
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    # Получаем confusion matrix: сколько и какие классы модель спутала
    matrix = confusion_matrix(all_targets, all_preds)
    # Вычисляем top-3 accuracy — насколько часто правильный класс входил в тройку наиболее вероятных
    top3_acc = top_k_accuracy_score(all_targets, probs_tensor.numpy(), k=3)

    print("\n=== Classification Report ===")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    print(f"Top-3 Accuracy: {top3_acc:.4f}")

    # Сохраняем подробные метрики по каждому классу в .csv для последующего анализа
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
    metrics_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })
    metrics_df.to_csv("results/detailed_metrics_arcface.csv", index=False)
    print("📄 Подробный отчёт сохранён в results/detailed_metrics_arcface.csv")

    return report, matrix, emb_tensor.numpy(), np.array(all_targets), top3_acc


# === Сохранение confusion matrix как картинки (визуализация) ===
def save_confusion_matrix(matrix, class_names, model_name):
    plt.figure(figsize=(10, 8))  # Размер графика
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказано')
    plt.ylabel('Истинно')
    plt.title(f'Матрица ошибок: {model_name}')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{model_name}_arcface.png')  # Сохраняем как изображение
    plt.close()


# === Сохраняем отчёт в .csv формате ===
def save_classification_report(report, model_name):
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'results/classification_report_{model_name}_arcface.csv', index=True)
    return df


# === Построение t-SNE проекции для эмбеддингов модели ===
def plot_tsne(embeddings, targets, class_names, model_name):
    print("🔍 Строим t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # Инициализация t-SNE
    reduced = tsne.fit_transform(embeddings)  # Понижаем размерность эмбеддингов до 2D
    df = pd.DataFrame(reduced, columns=['x', 'y'])  # Координаты точек
    df['label'] = [class_names[i] for i in targets]  # Подписываем точки классами

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='tab10', s=60)
    plt.title(f"t-SNE представление: {model_name} (ArcFace)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"results/tsne_{model_name}_arcface.png")
    plt.close()


# === Оценка модели по её названию (resnet50, efficientnet_b0, swin_tiny) ===
def evaluate_model(model_name):
    dataset_dir = "dataset_10classes"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = get_dataloaders(dataset_dir=dataset_dir, model_name=model_name)
    class_names = sorted(os.listdir(os.path.join(dataset_dir, 'test')))

    # Загружаем обученную модель и классификатор ArcFace
    features, arc_head, _ = get_model(model_name, arcface=True)
    checkpoint = torch.load(f'models/{model_name}_arcface.pth', map_location=device)
    features.load_state_dict(checkpoint['features_state'])
    arc_head.load_state_dict(checkpoint['arc_head_state'])
    features.to(device)
    arc_head.to(device)

    # Проводим оценку модели: метрики, визуализации и эмбеддинги
    report, matrix, embeddings, targets, top3_acc = evaluate(features, arc_head, test_loader, device, class_names)
    save_confusion_matrix(matrix, class_names, model_name)
    df = save_classification_report(report, model_name)

    # Добавляем Top-3 Accuracy как новую строку в таблицу
    df.loc["top3_accuracy"] = [None] * (df.shape[1] - 1) + [top3_acc]
    plot_tsne(embeddings, targets, class_names, model_name)

    # Считаем средний F1 только по настоящим классам (без accuracy и avg)
    class_rows = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    avg_f1 = df.loc[class_rows, 'f1-score'].mean().item()

    # Сохраняем подробный отчёт в Excel
    df.to_excel(f"results/classification_per_class_{model_name}_arcface.xlsx", index=True)
    return avg_f1


# === Функция для финального сравнения всех моделей по F1 и построения графика ===
def save_f1_summary(f1_scores):
    summary_df = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['avg_f1'])
    summary_df.to_excel("results/model_comparison_summary_arcface.xlsx")

    # Визуализируем сравнение моделей
    data = summary_df.reset_index().rename(columns={"index": "Model"})
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="Model", y="avg_f1", palette='viridis', hue="Model", dodge=False, legend=False)

    # Подписываем каждую колонку значением F1
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.ylabel("Средний F1-Score")
    plt.xlabel("Модель")
    plt.title("Сравнение моделей по F1 (ArcFace)", weight='bold')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/f1_score_comparison_arcface.png")
    plt.show()


# === Главная точка входа: запускается, если файл запущен напрямую ===
if __name__ == "__main__":
    models = ["resnet50", "efficientnet_b0", "swin_tiny"]  # Перечень моделей
    f1_scores = {}  # Словарь для хранения среднего F1 каждой модели

    for model_name in models:
        print(f"\n\ud83d\udd0d Оценка модели: {model_name}")
        avg_f1 = evaluate_model(model_name)
        f1_scores[model_name] = avg_f1

    # Сохраняем и визуализируем сравнение всех моделей
    save_f1_summary(f1_scores)
