#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from werkzeug.datastructures import FileStorage

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.backend.ml.emotions_recognitions_model import EmotionRecognitionModel
from src.backend.config import EMOTIONS

def test_emotion_recognition_one_trained():
    """
    Тест модели распознавания эмоций:
    1) Определяем аудиофайл для обучения и теста.
    2) Обучаем модель на заданной эмоции.
    3) Проверяем, что модель при предсказании возвращает ту же эмоцию.
    """
    # Пути к аудиофайлам и ожидаемая эмоция
    train_audio = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_Train_models.wav')
    test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_test_audioidentification.wav')
    expected_emotion = EMOTIONS[2]  # например, 'грусть'

    model = EmotionRecognitionModel()

    # Обучение модели
    with open(train_audio, 'rb') as f:
        storage = FileStorage(stream=f, filename=os.path.basename(train_audio), content_type='audio/wav')
        success = model.train(storage, expected_emotion)
        assert success, f"Обучение завершилось с ошибкой для эмоции '{expected_emotion}'"

    # Проверка предсказания
    with open(test_audio, 'rb') as f:
        storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
        predicted = model.predict(storage)
        assert predicted == expected_emotion, f"Ожидалось '{expected_emotion}', получено '{predicted}'"

def test_emotion_recognition_two_trained():
    pass
    # """
    # Тест модели распознавания эмоций:
    # 1) Определяем аудиофайл для обучения и теста.
    # 2) Обучаем модель на заданной эмоции.
    # 3) Проверяем, что модель при предсказании возвращает ту же эмоцию.
    # """
    # # Пути к аудиофайлам и ожидаемая эмоция
    # train_audio = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_Train_models.wav')
    # test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_test_audioidentification.wav')
    # expected_emotion = EMOTIONS[0]  # например, 'гнев'

    # model = EmotionRecognitionModel()

    # # Обучение модели
    # with open(train_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(train_audio), content_type='audio/wav')
    #     success = model.train(storage, expected_emotion)
    #     assert success, f"Обучение завершилось с ошибкой для эмоции '{expected_emotion}'"

    # # Проверка предсказания
    # with open(test_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
    #     predicted = model.predict(storage)
    #     assert predicted == expected_emotion, f"Ожидалось '{expected_emotion}', получено '{predicted}'"

def test_emotion_recognition_three_trained():
    pass
    # """
    # Тест модели распознавания эмоций:
    # 1) Определяем аудиофайл для обучения и теста.
    # 2) Обучаем модель на заданной эмоции.
    # 3) Проверяем, что модель при предсказании возвращает ту же эмоцию.
    # """
    # # Пути к аудиофайлам и ожидаемая эмоция
    # train_audio = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_Train_models.wav')
    # test_audio  = os.path.join(project_root, 'testing', 'test_input_audiofiles', 'For_test_audioidentification.wav')
    # expected_emotion = EMOTIONS[0]  # например, 'гнев'

    # model = EmotionRecognitionModel()

    # # Обучение модели
    # with open(train_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(train_audio), content_type='audio/wav')
    #     success = model.train(storage, expected_emotion)
    #     assert success, f"Обучение завершилось с ошибкой для эмоции '{expected_emotion}'"

    # # Проверка предсказания
    # with open(test_audio, 'rb') as f:
    #     storage = FileStorage(stream=f, filename=os.path.basename(test_audio), content_type='audio/wav')
    #     predicted = model.predict(storage)
    #     assert predicted == expected_emotion, f"Ожидалось '{expected_emotion}', получено '{predicted}'"

# Позволяет запустить тест вручную
if __name__ == "__main__":
    test_emotion_recognition_one_trained()
    test_emotion_recognition_two_trained()
    test_emotion_recognition_three_trained()