import numpy as np
from backend.api.error_logger import error_logger
from backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH

def ensure_min_length(audio_fragment):
    """
    Проверяет и обеспечивает минимальную длину аудиофрагмента.
    Если фрагмент слишком короткий, удлиняет его путем циклического повторения.
    
    Args:
        audio_fragment: numpy массив с аудиоданными
    
    Returns:
        numpy массив с гарантированной минимальной длиной
    """
    if audio_fragment is None or len(audio_fragment) == 0:
        error_logger.log_error("Пустой аудиофрагмент", "augmentation", "ensure_min_length")
        return np.zeros(int(AUDIO_FRAGMENT_LENGTH * SAMPLE_RATE))  # Возвращаем тишину минимальной длины
        
    min_length = int(AUDIO_FRAGMENT_LENGTH * SAMPLE_RATE)
    if len(audio_fragment) < min_length:
        # Используем библиотечный метод для повторения массива
        # np.tile выполняет эффективное циклическое повторение массива
        multiplier = int(np.ceil(min_length / len(audio_fragment)))
        extended_fragment = np.tile(audio_fragment, multiplier)[:min_length]
        
        error_logger.log_error(
            f"Аудиофрагмент удлинен с {len(audio_fragment)} до {len(extended_fragment)} отсчетов",
            "augmentation", "ensure_min_length"
        )
        
        return extended_fragment
    
    return audio_fragment 