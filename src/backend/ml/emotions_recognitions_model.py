import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import io
import random
from typing import List, Dict, Tuple, Union, Optional, Any, cast
from werkzeug.datastructures import FileStorage
from backend.api.error_logger import error_logger
from backend.api.info_logger import info_logger
from backend.config import EMOTIONS, SAMPLE_RATE, AUGMENTATION, MODELS_PARAMS
from backend.ml.audio_model_base import AudioModelBase

class EmotionRecognitionNN(nn.Module):
    """
    Нейронная сеть для распознавания эмоций в речи на основе PyTorch.
    Использует свёрточную архитектуру с дилатацией (аналог TDNN).
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Инициализация сети для распознавания эмоций
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (эмоций)
        """
        info_logger.info("---Start building EmotionRecognition model---")

        super(EmotionRecognitionNN, self).__init__()
        
        # Первый сверточный блок
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Второй сверточный блок с большей дилатацией
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Третий сверточный блок с ещё большей дилатацией
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Четвертый сверточный блок с шагом для уменьшения размерности
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Пятый сверточный блок для извлечения высокоуровневых признаков
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Полносвязный слой для классификации
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        info_logger.info("---Finish building EmotionRecognition model---")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть
        
        Args:
            x: Входные данные [batch_size, features, time]
            
        Returns:
            Предсказания модели
        """
        info_logger.info("---Start forwarding EmotionRecognition model---")

        # Сверточные блоки
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Глобальный пулинг
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Полносвязный слой
        x = self.fc(x)
        
        info_logger.info("---Finish forwarding EmotionRecognition model---")

        return x

class EmotionRecognitionModel(AudioModelBase[EmotionRecognitionNN]):
    """
    Модель для распознавания эмоций в речи.
    
    Атрибуты:
        model: Модель PyTorch для распознавания эмоций
        classes: Словарь эмоций и соответствующих им индексов
        index_to_emotion: Словарь индексов и соответствующих им эмоций
        device: Устройство для обучения модели (CPU/GPU)
    """
    
    def __init__(self) -> None:
        """
        Инициализирует модель для распознавания эмоций в речи.
        """
        super().__init__("emotions_recognition_model")
        
        # Инициализируем словари на основе списка эмоций из конфига
        for idx, emotion in enumerate(EMOTIONS):
            self.classes[emotion] = idx
            self.index_to_class[idx] = emotion
            
        self.index_to_emotion = self.index_to_class  # Для совместимости
            
    def _create_or_update_model(self, features: torch.Tensor) -> None:
        """
        Создает или обновляет модель для распознавания эмоций.
        
        Args:
            features: Тензор признаков для определения входной размерности
        """
        # Проверка, создана ли модель
        if self.model is None:
            info_logger.info("Creating new EmotionRecognition model")
            # Создаем новую модель
            input_dim: int = features.size(2)
            self.model = EmotionRecognitionNN(input_dim, len(self.classes)).to(self.device)
            info_logger.info("New EmotionRecognition model created")
    
    def predict(self, audio_file: FileStorage, expected_emotion: Optional[str] = None) -> Union[Dict[str, Union[str, float]], bool]:
        """
        Распознает эмоцию из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для распознавания
            expected_emotion: Ожидаемая эмоция для сравнения (опционально)
            
        Returns:
            dict: Результат распознавания с эмоцией и уверенностью
            или bool, если указана ожидаемая эмоция
        """
        info_logger.info("---Start prediction process in EmotionRecognition model---")
        try:
            # Используем базовый метод для извлечения признаков
            base_result = self._predict_base(audio_file)
            if base_result["status"] == "error":
                error_logger.log_error(
                    base_result["error_message"],
                    "emotions_recognition_model",
                    "predict"
                )
                if expected_emotion:
                    return False
                else:
                    return {"emotion": "unknown", "confidence": 0.0}
            
            features_list = base_result["features_list"]
            
            # Для прогнозирования используем только оригинальные признаки
            features: torch.Tensor = features_list[0]
            
            # Получаем предсказание модели
            prediction_result = self._get_prediction_from_model(features)
            confidence = prediction_result["confidence"]
            predicted_class_index_int = prediction_result["predicted_class_index"]
            
            # Если уверенность выше порога, распознаем эмоцию
            if confidence >= MODELS_PARAMS['MIN_CONFIDENCE']:
                # Получаем эмоцию по индексу класса
                if predicted_class_index_int in self.index_to_emotion:
                    emotion: str = self.index_to_emotion[predicted_class_index_int]
                else:
                    emotion = "unknown"
            else:
                emotion = "unknown"
            
            info_logger.info("End making prediction")
            
            # Если указана ожидаемая эмоция, сравниваем результат
            if expected_emotion:
                return emotion == expected_emotion and emotion != "unknown"
            
            # Иначе возвращаем распознанную эмоцию и уверенность
            return {"emotion": emotion, "confidence": confidence}
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "emotions_recognition_model",
                "predict",
                "Ошибка при распознавании эмоции"
            )
            
            if expected_emotion:
                return False
            else:
                return {"emotion": "unknown", "confidence": 0.0}
                
        finally:
            info_logger.info("---End prediction process in EmotionRecognition model---")
        
    def compare_emotion(self, audio_file: FileStorage, expected_emotion: str) -> bool:
        """
        Сравнивает эмоцию из аудиофайла с ожидаемой эмоцией.
        
        Args:
            audio_file: Аудиофайл для распознавания
            expected_emotion: Ожидаемая эмоция для сравнения
            
        Returns:
            bool: True, если эмоции совпадают, False в противном случае
        """
        info_logger.info("---Start emotion comparison process in EmotionRecognition model---")
        try:
            # Используем существующий метод predict с параметром expected_emotion
            result = cast(bool, self.predict(audio_file, expected_emotion))
            info_logger.info(f"Emotion comparison result: {result}")
            return result
        finally:
            info_logger.info("---End emotion comparison process in EmotionRecognition model---")
            
