import torch
import torch.nn as nn
from src.backend.config import EMOTIONS, EMOTIONS_MODEL_PARAMS
from src.backend.ml.common.base_model import BaseMLModel

class SqueezeExcitationBlock(nn.Module):
    """
    Блок Squeeze-and-Excitation для улучшения фильтрации каналов
    """
    def __init__(self, channel: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        # Squeeze операция
        y = self.avg_pool(x).view(b, c)
        # Excitation операция
        y = self.fc(y).view(b, c, 1)
        # Масштабирование исходных каналов
        return x * y.expand_as(x)

class EmotionRecognitionNN(nn.Module):
    """
    Нейронная сеть для распознавания эмоций в речи на основе PyTorch.
    Архитектура оптимизирована для предотвращения переобучения на малых наборах данных,
    но при этом достаточно мощная для хорошего разделения классов эмоций.
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Инициализация сети для распознавания эмоций
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (эмоций)
        """

        super(EmotionRecognitionNN, self).__init__()
        
        # Первый сверточный блок для извлечения основных признаков
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5)  # Уменьшаем dropout для первого слоя
        )
        
        # Второй сверточный блок с dilation
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(48),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        # Третий сверточный блок для дальнейшего извлечения признаков
        self.conv3 = nn.Sequential(
            nn.Conv1d(48, 72, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(72),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        # Блок Squeeze-and-Excitation для улучшения фильтрации каналов
        self.se_block = SqueezeExcitationBlock(72)
        
        # Механизм внимания для фокусировки на эмоционально значимых частях аудиосигнала
        self.attention = nn.Sequential(
            nn.Conv1d(72, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Полносвязные слои - сохраняем размерность 72 -> 32 -> num_classes
        self.fc = nn.Sequential(
            nn.Linear(72, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
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
        
        # Применяем Squeeze-and-Excitation
        x = self.se_block(x)
        
        # Применяем механизм внимания
        att_weights = self.attention(x)
        x = x * att_weights  # Поэлементное умножение для взвешивания признаков
        
        # Глобальный пулинг
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
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
    
