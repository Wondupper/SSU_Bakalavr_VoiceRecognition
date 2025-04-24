from backend.api.error_logger import error_logger
from backend.config import DATASET_CREATOR

# Используем константы из конфигурационного файла
EMOTIONS = DATASET_CREATOR['EMOTIONS']

def validate_audio_fragments(audio_fragments, context="common"):
    """
    Валидация аудиофрагментов перед созданием датасета
    
    Args:
        audio_fragments: Список аудиофрагментов для проверки
        context: Строка с контекстом валидации для сообщений об ошибках
        
    Returns:
        bool: True если данные валидны, иначе вызывает исключение
    """
    if not audio_fragments:
        error_msg = f"Нет аудиофрагментов для создания датасета ({context})"
        error_info = error_logger.log_exception(
            ValueError(error_msg),
            "validation.py",
            "validate_audio_fragments",
            "Проверка входных данных"
        )
        raise ValueError(error_msg)
    return True

def validate_emotion(emotion):
    """
    Валидация эмоции для создания датасета распознавания эмоций
    
    Args:
        emotion: Строка с названием эмоции
        
    Returns:
        bool: True если эмоция валидна, иначе вызывает исключение
    """
    if emotion not in EMOTIONS:
        error_msg = f"Недопустимая эмоция. Допустимые: {', '.join(EMOTIONS)}"
        error_info = error_logger.log_exception(
            ValueError(error_msg),
            "validation.py",
            "validate_emotion",
            "Проверка эмоции"
        )
        raise ValueError(error_msg)
    return True

def validate_dataset_result(dataset, context="common"):
    """
    Валидация результата создания датасета
    
    Args:
        dataset: Созданный датасет для проверки
        context: Строка с контекстом валидации для сообщений об ошибках
        
    Returns:
        bool: True если датасет валиден, иначе вызывает исключение
    """
    if not dataset:
        error_msg = f"Не удалось создать датасет: все операции извлечения признаков завершились с ошибками ({context})"
        error_info = error_logger.log_exception(
            ValueError(error_msg),
            "validation.py",
            "validate_dataset_result",
            "Проверка результатов обработки данных"
        )
        raise ValueError(error_msg)
    return True 