import os
import sys
from pathlib import Path
from werkzeug.datastructures import FileStorage

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.backend.ml.common.data_loader import load_voice_dataset
from src.backend.ml.voice_model import VoiceIdentificationModel


def test_voice_identification_trained():
    """ 
    Тест модели идентификации по голосу:
    1) Определяем аудиофайл для обучения и теста.
    2) Обучаем модель на заданном пользователе.
    3) Проверяем, что модель при предсказании возвращает то же имя пользователя.
    """
    voice_id_model: VoiceIdentificationModel = VoiceIdentificationModel()
    # Загрузка наборов данных
    voice_dataset = load_voice_dataset()
    # Обучение модели идентификации голоса
    voice_id_model.train(voice_dataset)
    test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'voice', 'i.wav')
    expected_user = 'unknown'

    # Проверка предсказания
    with open(test_audio, 'rb') as f:
        storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
        predicted = voice_id_model.predict(storage)
        assert predicted == expected_user, f"Ожидалось '{expected_user}', получено '{predicted}'"


# Позволяет запустить тест вручную
if __name__ == "__main__":
    test_voice_identification_trained()