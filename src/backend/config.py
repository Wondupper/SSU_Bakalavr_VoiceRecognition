"""
Конфигурационные параметры для backend части приложения
"""

from pathlib import Path

class AudioConfig:
    """Конфигурация для обработки аудио"""
    # Параметры для обработки аудио
    AUDIO_FRAGMENT_LENGTH = 3  # секунды
    SAMPLE_RATE = 16000  # Гц
    
    # Параметры для валидации входных данных
    MAX_AUDIOFILES_COUNT = 10
    MAX_AUDIOFILES_SIZE = 104857600  # 100 МБ в байтах
    INPUT_AUDIO_LENGTH = 5  # секунды
    INPUT_AUDIO_SIZE = 1048576  # 1 МБ в байтах
    ALLOWED_AUDIO_FORMAT = 'audio/wav'
    
    # Параметры для создания спектрограмм
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048

class ModelConfig:
    """Конфигурация для моделей"""
    # Общие параметры
    BATCH_SIZE = 32
    EPOCHS = 10
    VALIDATION_SPLIT = 0.2
    
    # Параметры для распознавания эмоций
    EMOTIONS = ['anger', 'joy', 'sadness']
    
    # Параметры для идентификации голоса
    UNKNOWN_USER = 'unknown'

class PathConfig:
    """Конфигурация путей для сохранения данных"""
    # Базовая директория проекта
    BASE_DIR = Path(__file__).parent.parent
    
    # Директории для данных
    MODELS_DIR = BASE_DIR / 'models'  # Для сохранения обученных моделей
    TEMP_DIR = BASE_DIR / 'temp'      # Для временных файлов
    
    # Создание директорий при импорте
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True) 