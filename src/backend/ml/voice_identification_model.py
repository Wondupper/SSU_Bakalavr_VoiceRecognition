import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import random
import io
from typing import List, Dict, Tuple, Union, Optional, Any, Set
from werkzeug.datastructures import FileStorage
from backend.api.error_logger import error_logger
from backend.api.info_logger import info_logger
from backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH, AUGMENTATION, MODELS_PARAMS
from backend.ml.audio_model_base import AudioModelBase

class VoiceIdentificationNN(nn.Module):
    """
    Нейронная сеть для идентификации голоса на основе PyTorch.
    Использует свёрточную архитектуру с дилатацией.
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Инициализация сети для идентификации по голосу
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (пользователей)
        """
        info_logger.info("---Start building VoiceIdentification model---")
        super(VoiceIdentificationNN, self).__init__()
        
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
        info_logger.info("---Finish building VoiceIdentification model---")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть
        
        Args:
            x: Входные данные [batch_size, features, time]
            
        Returns:
            Предсказания модели
        """
        info_logger.info("---Start forwarding VoiceIdentification model---")
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
        info_logger.info("---Finish forwarding VoiceIdentification model---")
        return x

class VoiceIdentificationModel(AudioModelBase[VoiceIdentificationNN]):
    """
    Модель для идентификации пользователя по голосу.
    
    Атрибуты:
        model: Модель PyTorch для идентификации по голосу
        classes: Словарь имен пользователей и соответствующих им индексов
        index_to_name: Словарь индексов и соответствующих им имен пользователей
        device: Устройство для обучения модели (CPU/GPU)
    """
    
    def __init__(self) -> None:
        """
        Инициализирует модель для идентификации по голосу.
        """
        super().__init__("voice_identification")
        self.index_to_name = self.index_to_class  # Для совместимости
    
    def _create_or_update_model(self, features: torch.Tensor) -> None:
        """
        Создает или обновляет модель для идентификации голоса.
        
        Args:
            features: Тензор признаков для определения входной размерности
        """
        # Проверка, создана ли модель и соответствует ли она текущим классам
        if self.model is None or self.model.fc[-1].out_features != len(self.classes):
            info_logger.info("Creating new VoiceIdentification model")
            # Создаем новую модель
            input_dim: int = features.size(2)
            self.model = VoiceIdentificationNN(input_dim, len(self.classes)).to(self.device)
            info_logger.info("New VoiceIdentification model created")
    
    def predict(self, audio_file: FileStorage) -> str:
        """
        Идентифицирует пользователя по голосу из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для идентификации
            
        Returns:
            str: Имя пользователя или "unknown", если не удалось идентифицировать
        """
        info_logger.info("---Start prediction process in VoiceIdentification model---")
        try:
            # Используем базовый метод для извлечения признаков
            base_result = self._predict_base(audio_file)
            if base_result["status"] == "error":
                error_logger.log_error(
                    base_result["error_message"],
                    "voice_identification",
                    "predict"
                )
                return "unknown"
            
            features_list = base_result["features_list"]
            
            # Преобразуем в тензоры PyTorch
            info_logger.info("Start converting features to PyTorch tensors")
            X: torch.Tensor = torch.stack(features_list).to(self.device)
            info_logger.info("End converting features to PyTorch tensors")
            
            # Предсказание
            info_logger.info("Start making prediction")
            
            # Голосование по результатам фрагментов
            vote_counts: Dict[str, int] = {}
            confidence_sums: Dict[str, float] = {}
            
            with torch.no_grad():
                for feature in X:
                    # Получаем предсказание для каждого фрагмента
                    prediction_result = self._get_prediction_from_model(feature)
                    confidence = prediction_result["confidence"]
                    predicted_class_index_int = prediction_result["predicted_class_index"]
                    
                    # Если уверенность выше порога, идентифицируем пользователя
                    if confidence >= MODELS_PARAMS['MIN_CONFIDENCE']:
                        # Получаем имя пользователя по индексу класса
                        if predicted_class_index_int in self.index_to_name:
                            label: str = self.index_to_name[predicted_class_index_int]
                        else:
                            label = "unknown"
                    else:
                        label = "unknown"
                    
                    if label not in vote_counts:
                        vote_counts[label] = 0
                        confidence_sums[label] = 0.0
                    
                    vote_counts[label] += 1
                    confidence_sums[label] += confidence
            
            # Находим метку с наибольшим количеством голосов
            if vote_counts:
                identity: str = max(vote_counts.keys(), key=lambda k: vote_counts[k])
                avg_confidence: float = confidence_sums[identity] / vote_counts[identity]
                
                # Если средняя уверенность низкая или это "unknown", считаем неизвестным
                if avg_confidence < MODELS_PARAMS['MIN_AVG_CONFIDENCE'] or identity == "unknown":
                    info_logger.info("Prediction result: unknown (low confidence)")
                    return "unknown"
                    
                info_logger.info(f"Prediction result: {identity} (confidence: {avg_confidence:.4f})")
                return identity
            else:
                info_logger.info("Prediction result: unknown (no votes)")
                return "unknown"
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "predict",
                "Ошибка при идентификации пользователя"
            )
            
            return "unknown"
            
        finally:
            info_logger.info("---End prediction process in VoiceIdentification model---")
