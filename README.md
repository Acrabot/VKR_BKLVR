# ВКР: Классификация изображений минералов с использованием нейросетевых моделей

**Репозиторий содержит исходный код дипломной работы по теме:  
«Исследование нейросетевых технологий для классификации минералов»**

---

## 🎯 Цель работы

Провести сравнительный анализ трёх современных нейросетевых моделей — **ResNet-50**, **EfficientNet-B0** и **Swin Transformer-Tiny** — в задаче многоклассовой классификации изображений минералов, используя открытый датасет **MineralImage5k**.

---

## 🧠 Архитектуры моделей

- `ResNet-50` — сверточная сеть с остаточными связями ([2])
- `EfficientNet-B0` — сбалансированная модель по параметрам FLOPs / точность ([3])
- `Swin Transformer-Tiny` — визуальный трансформер с оконным сдвигом ([4])

Все модели использовались с **ArcFace-классификатором** для повышения межклассовой разделимости.

---

## 📂 Структура проекта
├── main.py # Объединённый запуск обучения и оценки
├── train.py # Цикл обучения с LLRD, AMP, scheduler, ArcFace
├── evaluate.py # Генерация метрик, confusion matrix, t-SNE
├── model.py # Модели: ResNet, EfficientNet, Swin, ArcMarginProduct
├── dataloader.py # Загрузка датасета и аугментации
├── prepare_dataset.py # Очистка и сборка датасета MineralImage5k
├── results/ # Результаты: метрики, графики, таблицы
├── models/ # Чекпойнты обученных моделей
├── LICENSE # MIT License
└── README.md # Описание проекта

---

## 🧪 Метрики и визуализация

Оценка проводилась по следующим показателям:
- **Accuracy**
- **Macro F1-score**
- **Top-3 Accuracy**
- Матрицы ошибок (confusion matrix)
- Представления t-SNE по эмбеддингам

Все результаты сохраняются в `results/`:
- `.csv` и `.xlsx` — по классам
- `.png` — графики сравнения, confusion matrix, t-SNE

---

## 💾 Используемый датасет

- 📦 **MineralImage5k**  
- Источник: [https://github.com/ai-forever/mineral-recognition](https://disk.yandex.ru/d/KapicF_MEysifg)  
- Поддержка ограниченного набора из 10 классов
- Картинки очищены и нормализованы до 256×256 px  
- Используются bounding boxes с фильтрацией и CLAHE (контраст)

---

## ⚙️ Запуск проекта

### Обучение всех моделей
  python main.py
### Отдельно: обучение
  python train.py
###Отдельно: оценка
  python evaluate.py
  
⚠️ Перед запуском проверь, что директория dataset_10classes содержит готовые выборки (train / val / test)

---

📦 Зависимости (Python ≥ 3.10)
torch, torchvision
scikit-learn
pandas, numpy
matplotlib, seaborn
tqdm
opencv-python
Pillow
Unidecode

---

📜 Лицензия
Проект распространяется под лицензией MIT License.

---

👤 Автор
Артём, студент Санкт-Петербургского горного университета
Направление подготовки: 09.03.02 «Информационные системы и технологии»
Год выполнения ВКР: 2025

---

🔗 Цитирование ключевых источников
[2] He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition // CVPR. 2016.
[3] Tan M., Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks // ICML. 2019.
[4] Liu Z. et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows // ICCV. 2021.
