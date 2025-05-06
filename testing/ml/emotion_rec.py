import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from werkzeug.datastructures import FileStorage
from src.backend.ml.common.data_loader import load_emotions_dataset
from src.backend.ml.emotions_model import EmotionRecognitionModel
from src.backend.config import EMOTIONS



def test_emotion_recognition_trained():
    """
    Тест модели распознавания эмоций:
    1) Определяем аудиофайл для обучения и теста.
    2) Обучаем модель на заданной эмоции.
    3) Проверяем, что модель при предсказании возвращает ту же эмоцию.
    """
    emotion_model: EmotionRecognitionModel = EmotionRecognitionModel()
    emotions_dataset = load_emotions_dataset()
    # Обучение модели распознавания эмоций
    emotion_model.train(emotions_dataset)
    # Пути к аудиофайлам и ожидаемая эмоция
    test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'voice', 'secondspeaker10sec.wav')
    expected_emotion = EMOTIONS[1]  # например, 'спокойствие'

    # Проверка предсказания - используем уже обученную модель
    with open(test_audio, 'rb') as f:
        storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
        predicted = emotion_model.predict(storage)
        assert predicted == expected_emotion, f"Ожидалось {expected_emotion}, получено {predicted}"

# Позволяет запустить тест вручную
if __name__ == "__main__":
    test_emotion_recognition_trained()