import librosa
from backend.api.error_logger import error_logger

# Импортируем функции из созданных модулей
from backend.processors.augmentation_processors.other.validator import ensure_min_length
from backend.processors.augmentation_processors.other.noise_remove import remove_noise

def augment_audio(audio_fragments):
    """
    Аугментация аудиофрагментов по обновленной спецификации
    с оптимизацией для снижения расхода памяти
    
    Спецификация аугментации:
    1. Группа A: Шумоподавление (1 операция)
    2. Группа B1: Замедление записи в 0.8 раза (1 операция)
    3. Группа B2: Ускорение записи в 1.3 раза (1 операция)
    
    Процесс применения:
    1. Пункт 1: Удваиваем исходный набор удалением шума (группа A)
    2. Пункт 2: Расширяем набор из пункта 1 замедлением (группа B1)
    3. Пункт 3: Расширяем набор из пункта 1 ускорением (группа B2)
    
    Это должно увеличить размер датасета в 6 раз.
    """
    if not audio_fragments:
        return []
        
    # Предварительно проверяем и удлиняем слишком короткие фрагменты
    audio_fragments = [ensure_min_length(fragment) for fragment in audio_fragments]
        
    # Шаг 1: Сохраняем исходные фрагменты
    original_fragments = audio_fragments.copy()
    
    # Группа A: Удаление шума (Пункт 1)
    # Применяем ко всем исходным фрагментам
    denoised_fragments = []
    for fragment in original_fragments:
        try:
            denoised = remove_noise(fragment)
            # Проверяем длину после шумоподавления
            denoised = ensure_min_length(denoised)
            denoised_fragments.append(denoised)
        except Exception as e:
            error_logger.log_error(f"Ошибка удаления шума: {str(e)}", "augmentation", "augment_audio")
    
    # Объединяем исходные и обработанные фрагменты (Пункт 1 в спецификации)
    step1_fragments = original_fragments.copy() + denoised_fragments
    
    # Группа B1: Замедление записи в 0.8 раза (Пункт 2)
    # Применяем к результатам пункта 1
    slowdown_fragments = []
    speed_factor_slow = 0.8  # Только один фактор замедления
    
    for fragment in step1_fragments:
        try:
            # Используем librosa.effects.time_stretch вместо самописной функции
            slow_fragment = librosa.effects.time_stretch(fragment, rate=speed_factor_slow)
            # Проверяем длину после изменения скорости
            slow_fragment = ensure_min_length(slow_fragment)
            slowdown_fragments.append(slow_fragment)
        except Exception as e:
            error_logger.log_error(f"Ошибка замедления: {str(e)}", "augmentation", "augment_audio")
    
    # Группа B2: Ускорение записи в 1.3 раза (Пункт 3)
    # Применяем к результатам пункта 1
    speedup_fragments = []
    speed_factor_fast = 1.3  # Только один фактор ускорения
    
    for fragment in step1_fragments:
        try:
            # Используем librosa.effects.time_stretch вместо самописной функции
            fast_fragment = librosa.effects.time_stretch(fragment, rate=speed_factor_fast)
            # Проверяем длину после изменения скорости
            fast_fragment = ensure_min_length(fast_fragment)
            speedup_fragments.append(fast_fragment)
        except Exception as e:
            error_logger.log_error(f"Ошибка ускорения: {str(e)}", "augmentation", "augment_audio")
    
    # Собираем все фрагменты
    result_fragments = []
    # Добавляем оригинальные фрагменты
    result_fragments.extend(original_fragments)
    # Добавляем фрагменты с удаленным шумом
    result_fragments.extend(denoised_fragments)
    # Добавляем замедленные фрагменты
    result_fragments.extend(slowdown_fragments)
    # Добавляем ускоренные фрагменты
    result_fragments.extend(speedup_fragments)
        
    return result_fragments
