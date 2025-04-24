import numpy as np
from backend.api.error_logger import error_logger
from backend.config import AUDIO_FRAGMENT_LENGTH

def split_audio_into_fixed_length_segments(audio_data, sr):
    """
    Разделение аудио на фрагменты фиксированной длины.
    Упрощенная и оптимизированная версия с использованием numpy.
    """
    if len(audio_data) == 0:
        error_logger.log_exception(
            ValueError("Пустые аудиоданные"),
            "audio_processing",
            "audio_splitting",
            "Проверка входных данных"
        )
        return []
    
    try:
        # Размер одного фрагмента в отсчетах
        fragment_size = int(AUDIO_FRAGMENT_LENGTH * sr)
        min_fragment_size = int(fragment_size * 0.25)  # Минимальный размер (25% от полного)
        
        # Проверка минимальной длины аудио
        if len(audio_data) < min_fragment_size:
            error_logger.log_error(
                f"Аудио слишком короткое: {len(audio_data)} отсчетов, минимум: {min_fragment_size}",
                "audio", "split_audio_into_fixed_length_segments"
            )
            return []
        
        # Для коротких аудио: дополнение нулями до полного фрагмента
        if len(audio_data) < fragment_size:
            fragment = np.zeros(fragment_size, dtype=np.float32)
            fragment[:len(audio_data)] = audio_data
            return [fragment]
        
        # Определяем количество полных фрагментов
        num_fragments = len(audio_data) // fragment_size
        fragments = []
        
        # Эффективное разделение с помощью numpy.array_split
        if num_fragments > 0:
            # Создаем полные фрагменты
            full_fragments_data = audio_data[:num_fragments * fragment_size]
            fragments = np.array_split(full_fragments_data, num_fragments)
            
            # Преобразуем массивы numpy в список и делаем копии для предотвращения проблем с памятью
            fragments = [fragment.copy() for fragment in fragments]
        
        # Обработка остатка, если он достаточно большой
        remainder = len(audio_data) % fragment_size
        if remainder >= min_fragment_size:
            # Создаем дополненный нулями фрагмент
            last_fragment = np.zeros(fragment_size, dtype=np.float32)
            last_fragment[:remainder] = audio_data[-remainder:].copy()
            fragments.append(last_fragment)
        
        return fragments
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_processing",
            "audio_splitting",
            "Ошибка при разделении аудио на фрагменты"
        )
        return []

def create_default_fragment(audio_data, sr):
    """
    Создает один фрагмент фиксированной длины из аудиоданных
    """
    fragment_size = int(AUDIO_FRAGMENT_LENGTH * sr)
    fragment = np.zeros(fragment_size, dtype=np.float32)
    copy_size = min(len(audio_data), fragment_size)
    fragment[:copy_size] = audio_data[:copy_size]
    return [fragment] 