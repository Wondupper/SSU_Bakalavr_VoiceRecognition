import torch
import torch.nn as nn
from src.backend.config import EMOTIONS, EMOTIONS_MODEL_PARAMS
from src.backend.ml.common.base_model import BaseMLModel
from werkzeug.datastructures import FileStorage
from src.backend.ml.common.audio_to_features import get_features_tensors_from_audio_for_training
from src.backend.loggers.error_logger import error_logger

class EmotionRecognitionNN(nn.Module):
    """
    Нейронная сеть для распознавания эмоций в речи на основе PyTorch.
    Архитектура упрощена для предотвращения переобучения на малых наборах данных.
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Инициализация сети для распознавания эмоций
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (эмоций)
        """

        super(EmotionRecognitionNN, self).__init__()
        
        # Уменьшаем количество фильтров и добавляем больше Dropout
        # Первый сверточный блок с сильной регуляризацией
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)  # Увеличенный дропаут
        )
        
        # Второй сверточный блок с Max Pooling для уменьшения размерности
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.4)  # Увеличенный дропаут
        )
        
        # Третий сверточный блок - меньше фильтров, чем было раньше
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.4)  # Увеличенный дропаут
        )
        
        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Полносвязный слой для классификации - упрощен
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # Высокий уровень дропаута для предотвращения переобучения
            nn.Linear(32, num_classes)
        )
    

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть
        
        Args:
            x: Входные данные [batch_size, features, time] или [batch_size, time, features]
            
        Returns:
            Предсказания модели
        """
        
        # Безопасное преобразование формата данных
        # Проверка на правильность размерностей
        if len(x.shape) >= 3:
            # Если данные приходят в формате [batch_size, time, features], 
            # преобразуем их в формат [batch_size, features, time]
            if x.shape[1] > x.shape[2]:
                x = x.transpose(1, 2)
        else:
            # Если размерность не 3D, добавляем размерность батча
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
        
        # Проверка размерности после преобразования
        if len(x.shape) != 3:
            raise ValueError(f"Некорректная размерность после преобразования: {x.shape}, ожидается 3D тензор")

        # Сверточные блоки
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Глобальный пулинг
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Полносвязный слой
        x = self.fc(x)

        return x

class EmotionRecognitionModel(BaseMLModel):
    """
    Модель для распознавания эмоций в речи.
    """
    
    def __init__(self) -> None:
        """
        Инициализирует модель для распознавания эмоций в речи.
        """
        super().__init__("emotions_recognitions_model", EMOTIONS_MODEL_PARAMS)
        
        # Инициализируем словари на основе списка эмоций из конфига
        for idx, emotion in enumerate(EMOTIONS):
            self.classes[emotion] = idx
            self.index_to_class[idx] = emotion
        
    def create_model(self, input_dim: int, num_classes: int) -> nn.Module:
        """
        Создает модель нейронной сети для распознавания эмоций.
        
        Args:
            input_dim: Размерность входных данных
            num_classes: Количество классов
            
        Returns:
            nn.Module: Модель нейронной сети
        """
        return EmotionRecognitionNN(input_dim, num_classes)
    
