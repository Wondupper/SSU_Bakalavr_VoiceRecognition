import torch
import torch.nn as nn
from werkzeug.datastructures import FileStorage
from backend.config import EMOTIONS
from backend.ml.audio_model_base import AudioModelBase

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
            x: Входные данные [batch_size, features, time]
            
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
    
    def predict(self, audio_file: FileStorage) -> str:
        """
        Распознает эмоцию из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для распознавания
            
        Returns:
            str: Предсказанная эмоция
        """
        return super().predict(audio_file)
                
