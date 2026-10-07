# Классификация животных по фотографиям (ResNet50)

Нейросеть, которая по фотографии определяет одно из **10 животных**:
белка, бабочка, кошка, курица, корова, собака, слон, лошадь, овца, паук.

## Подход

- **Данные:** около 26 000 изображений 10 классов (датасет в духе [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)), папка [`processed_img/`](processed_img). Данные разбиты на обучающую и валидационную выборки (85 / 15 %).
- **Аугментация:** повороты, сдвиги, масштабирование и горизонтальные отражения через `ImageDataGenerator`.
- **Модель:** transfer learning. Свёрточная часть **ResNet50**, предобученная на ImageNet, заморожена. Поверх неё добавлены полносвязные слои `64 → 64 → 128 → 10` с Dropout 0.3.
  Всего 23.7 млн параметров, из них обучаемых около 145 тыс.
- **Результат:** точность на валидации около **75–79 %** уже после 10 эпох обучения «головы» сети.

## Файлы

| Путь | Описание |
|------|----------|
| [`test_3.ipynb`](test_3.ipynb) | Ноутбук: загрузка данных, обучение, графики, матрица ошибок, предсказания |
| `model_res50.h5` | Обученная модель (Keras, ~96 МБ) |
| `processed_img/` | Обучающая выборка по папкам-классам |
| `Predict/` | Изображения для проверки предсказаний |
| `USER_Animal/` | Пользовательские фото для демонстрации |

## Запуск

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
jupyter notebook test_3.ipynb
```

Пример предсказания на готовой модели:

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

labels = ['belka', 'butterfly', 'cats', 'chicken', 'cows',
          'dogs', 'elephant', 'horses', 'sheep', 'spider']

model = keras.models.load_model('model_res50.h5')
img = image.load_img('USER_Animal/1.jpeg', target_size=(300, 300))
x = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
print(labels[np.argmax(model.predict(x))])
```

## Что можно улучшить

- разморозить верхние блоки ResNet50 (fine-tuning) и обучать дольше;
- попробовать EfficientNet / ConvNeXt;
- сделать веб-демо (Gradio / Streamlit).

## Стек

Python, TensorFlow / Keras, NumPy, Matplotlib, Seaborn, scikit-learn
