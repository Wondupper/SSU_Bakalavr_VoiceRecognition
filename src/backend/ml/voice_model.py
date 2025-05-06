import torch
import torch.nn as nn
from src.backend.config import VOICE_MODEL_PARAMS
from src.backend.ml.common.base_model import BaseMLModel
from werkzeug.datastructures import FileStorage
from src.backend.ml.common.audio_to_features import get_features_tensors_from_audio_for_training

class VoiceIdentificationNN(nn.Module):
    """
    Нейронная сеть для идентификации голоса на основе PyTorch.
    Архитектура упрощена для предотвращения переобучения на малых наборах данных.
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Инициализация сети для идентификации по голосу
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (пользователей)
        """
        super(VoiceIdentificationNN, self).__init__()
        
        # Упрощаем архитектуру - меньше фильтров и больше регуляризации
        # Первый сверточный блок
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)  # Увеличенный дропаут
        )
        
        # Второй сверточный блок с Max Pooling для уменьшения размерности
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(48),
            nn.MaxPool1d(2),
            nn.Dropout(0.4)  # Увеличенный дропаут
        )
        
        # Третий сверточный блок
        self.conv3 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, stride=1, padding=1),
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
            nn.Dropout(0.5),  # Высокий дропаут для предотвращения переобучения
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

class VoiceIdentificationModel(BaseMLModel):
    """
    Модель для идентификации пользователя по голосу.
    """
    
    def __init__(self) -> None:
        """
        Инициализирует модель для идентификации по голосу.
        """
        super().__init__("voice_identification_model", VOICE_MODEL_PARAMS)
        
    def create_model(self, input_dim: int, num_classes: int) -> nn.Module:
        """
        Создает модель нейронной сети для идентификации по голосу.
        
        Args:
            input_dim: Размерность входных данных
            num_classes: Количество классов
            
        Returns:
            nn.Module: Модель нейронной сети
        """
        return VoiceIdentificationNN(input_dim, num_classes)
    
    
