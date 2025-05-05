import os
import sys
from pathlib import Path
from werkzeug.datastructures import FileStorage

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.backend.ml.voice_identification.voice_identification_model import VoiceIdentificationModel


def test_voice_identification_one_trained():
    """
    Тест модели идентификации по голосу:
    1) Определяем аудиофайл для обучения и теста.
    2) Обучаем модель на заданном пользователе.
    3) Проверяем, что модель при предсказании возвращает то же имя пользователя.
    """
    # Пути к аудиофайлам и ожидаемое имя пользователя
    train_audio = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_Train_models.wav')
    test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_test_audioidentification.wav')
    expected_user = 'testuser'

    model = VoiceIdentificationModel()

    # Обучение модели
    with open(train_audio, 'rb') as f:
        storage = FileStorage(stream=f, filename=os.path.basename(train_audio), content_type='audio/wav')
        success = model.train(storage, expected_user)
        assert success, f"Обучение завершилось с ошибкой для пользователя '{expected_user}'"

    # Проверка предсказания
    with open(test_audio, 'rb') as f:
        storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
        predicted = model.predict(storage)
        assert predicted == expected_user, f"Ожидалось '{expected_user}', получено '{predicted}'"

def test_voice_identification_two_trained():
    pass
    # """
    # Тест модели идентификации по голосу:
    # 1) Определяем аудиофайл для обучения и теста.
    # 2) Обучаем модель на заданном пользователе.
    # 3) Проверяем, что модель при предсказании возвращает то же имя пользователя.
    # """
    # # Пути к аудиофайлам и ожидаемое имя пользователя
    # train_audio = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_Train_models.wav')
    # test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_test_audioidentification.wav')
    # expected_user = 'testuser'

    # model = VoiceIdentificationModel()

    # # Обучение модели
    # with open(train_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(train_audio), content_type='audio/wav')
    #     success = model.train(storage, expected_user)
    #     assert success, f"Обучение завершилось с ошибкой для пользователя '{expected_user}'"

    # # Проверка предсказания
    # with open(test_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
    #     predicted = model.predict(storage)
    #     assert predicted == expected_user, f"Ожидалось '{expected_user}', получено '{predicted}'"

def test_voice_identification_three_trained():
    pass
    # """
    # Тест модели идентификации по голосу:
    # 1) Определяем аудиофайл для обучения и теста.
    # 2) Обучаем модель на заданном пользователе.
    # 3) Проверяем, что модель при предсказании возвращает то же имя пользователя.
    # """
    # # Пути к аудиофайлам и ожидаемое имя пользователя
    # train_audio = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_Train_models.wav')
    # test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_test_audioidentification.wav')
    # expected_user = 'testuser'

    # model = VoiceIdentificationModel()

    # # Обучение модели
    # with open(train_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(train_audio), content_type='audio/wav')
    #     success = model.train(storage, expected_user)
    #     assert success, f"Обучение завершилось с ошибкой для пользователя '{expected_user}'"

    # # Проверка предсказания
    # with open(test_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
    #     predicted = model.predict(storage)
    #     assert predicted == expected_user, f"Ожидалось '{expected_user}', получено '{predicted}'"

# Позволяет запустить тест вручную
if __name__ == "__main__":
    test_voice_identification_one_trained()
    test_voice_identification_two_trained()
    test_voice_identification_three_trained()