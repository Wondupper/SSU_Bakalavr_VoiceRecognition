import torch
import torch.nn as nn
from werkzeug.datastructures import FileStorage
from backend.ml.audio_model_base import AudioModelBase

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
        
        # Если данные приходят в формате [batch_size, time, features], 
        # преобразуем их в формат [batch_size, features, time]
        if x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)
            
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
     
    def predict(self, audio_file: FileStorage) -> str:
        """
        Идентифицирует пользователя по голосу из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для идентификации
            
        Returns:
            str: Имя пользователя или "unknown", если не удалось идентифицировать
        """
        return super().predict_extended(audio_file)
