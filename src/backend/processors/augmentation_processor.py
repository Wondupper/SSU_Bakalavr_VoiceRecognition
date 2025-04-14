import numpy as np
import librosa
import random
from functools import lru_cache  # Добавлен импорт для кэширования
from concurrent.futures import ProcessPoolExecutor  # Поддержка многопроцессорности
import multiprocessing
from backend.api.error_logger import error_logger
from functools import partial
import sys
import os
import scipy.signal
import gc

# Константы для оптимизации
MAX_AUGMENTED_SAMPLES = 1000  # Уменьшено с 5000 до 1000, так как теперь набор увеличивается в 12 раз вместо 90
N_FFT = 1024         # Размер окна для БПФ (уменьшен с 2048 для предотвращения ошибок памяти)
HOP_LENGTH = 512

# Определяем оптимальное количество процессов
# Ограничиваем максимальное количество процессоров для снижения нагрузки на память
N_JOBS = max(1, min(2, multiprocessing.cpu_count() - 1))  # Не более 2 процессов

# Константы для работы с аудио и аугментации
SAMPLE_RATE = 16000  # Стандартная частота дискретизации
MIN_SNR_DB = 5       # Минимальное отношение сигнал/шум для добавления шума
MAX_SNR_DB = 15      # Максимальное отношение сигнал/шум для добавления шума
MAX_PITCH_SHIFT = 2  # Максимальное изменение высоты тона (в полутонах)
MIN_SPEED = 0.7      # Минимальный коэффициент изменения скорости
MAX_SPEED = 1.5      # Максимальный коэффициент изменения скорости
# Ограничение по размеру батча для экономии памяти
MAX_BATCH_SIZE = 8   # Максимальный размер батча для параллельной обработки

# Параметры временного маскирования
MASK_COUNT_MIN = 2   # Минимальное количество масок
MASK_COUNT_MAX = 5   # Максимальное количество масок
MASK_LENGTH_MIN = 0.05  # Минимальная длина маски в секундах
MASK_LENGTH_MAX = 0.15  # Максимальная длина маски в секундах

# Минимальная длина аудиофрагмента, необходимая для обработки
MIN_AUDIO_LENGTH = 4096  # Равно N_FFT * 2 из dataset_creator.py

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
        return np.zeros(MIN_AUDIO_LENGTH)  # Возвращаем тишину минимальной длины
        
    if len(audio_fragment) < MIN_AUDIO_LENGTH:
        # Используем циклическое повторение (вместо простого дополнения нулями)
        # для сохранения характеристик сигнала
        multiplier = int(np.ceil(MIN_AUDIO_LENGTH / len(audio_fragment)))
        extended_fragment = np.tile(audio_fragment, multiplier)[:MIN_AUDIO_LENGTH]
        
        error_logger.log_error(
            f"Аудиофрагмент удлинен с {len(audio_fragment)} до {len(extended_fragment)} отсчетов",
            "augmentation", "ensure_min_length"
        )
        
        return extended_fragment
    
    return audio_fragment

def apply_time_masking(audio_data, sr=SAMPLE_RATE):
    """
    Применяет временное маскирование к аудиофрагменту, заглушая случайные 
    сегменты для увеличения устойчивости модели к потере частей информации.
    
    Args:
        audio_data: numpy массив с аудиоданными
        sr: частота дискретизации аудио
        
    Returns:
        numpy массив с замаскированными сегментами
    """
    try:
        # Проверка на пустые данные
        if audio_data is None or len(audio_data) == 0:
            return audio_data
            
        # Создаем копию аудио для модификации
        result = np.copy(audio_data)
        
        # Определяем количество масок
        mask_count = np.random.randint(MASK_COUNT_MIN, MASK_COUNT_MAX + 1)
        
        # Длина аудио в отсчетах
        audio_length = len(audio_data)
        
        # Минимальная и максимальная длина маски в отсчетах
        min_mask_length = int(sr * MASK_LENGTH_MIN)
        max_mask_length = int(sr * MASK_LENGTH_MAX)
        
        # Для очень коротких аудио корректируем длину маски
        if max_mask_length > audio_length // 4:
            max_mask_length = audio_length // 4
            min_mask_length = min(min_mask_length, max_mask_length // 2)
        
        # Применяем маски
        for _ in range(mask_count):
            # Определяем длину текущей маски
            mask_length = np.random.randint(min_mask_length, max_mask_length + 1)
            
            # Определяем начало маски (не ближе 10% к началу и концу)
            safe_margin = int(audio_length * 0.1)
            mask_start = np.random.randint(
                safe_margin, 
                audio_length - mask_length - safe_margin
            )
            
            # Заглушаем сегмент (заменяем на тишину)
            result[mask_start:mask_start + mask_length] = 0
        
        return result
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            
        error_logger.log_error(
            f"Ошибка при применении временного маскирования: {str(e)}", 
            "augmentation", "apply_time_masking"
        )
        return audio_data  # В случае ошибки возвращаем исходный аудиофрагмент

def augment_audio(audio_fragments):
    """
    Аугментация аудиофрагментов по обновленной спецификации
    с оптимизацией для снижения расхода памяти
    
    Спецификация аугментации:
    1. Группа A: Шумоподавление (1 операция)
    2. Группа B1: Замедление записи в 0.8 раза (1 операция)
    3. Группа B2: Ускорение записи в 1.3 раза (1 операция)
    4. Группа C: Временное маскирование (1 операция)
    
    Процесс применения:
    1. Пункт 1: Удваиваем исходный набор удалением шума (группа A)
    2. Пункт 2: Расширяем набор из пункта 1 замедлением (группа B1)
    3. Пункт 3: Расширяем набор из пункта 1 ускорением (группа B2)
    4. Пункт 4: Применяем временное маскирование ко всем результатам из пунктов 2 и 3
    
    Это должно увеличить размер датасета в 12 раз.
    """
    if not audio_fragments:
        return []
        
    # Предварительно проверяем и удлиняем слишком короткие фрагменты
    audio_fragments = [ensure_min_length(fragment) for fragment in audio_fragments]
        
    # Ограничим исходное количество фрагментов для экономии памяти
    MAX_ORIGINAL_FRAGMENTS = 5  # Устанавливаем максимум исходных фрагментов
    if len(audio_fragments) > MAX_ORIGINAL_FRAGMENTS:
        import random
        audio_fragments = random.sample(audio_fragments, MAX_ORIGINAL_FRAGMENTS)
        error_logger.log_error(
            f"Количество исходных фрагментов сокращено до {MAX_ORIGINAL_FRAGMENTS} для экономии памяти",
            "augmentation", "augment_audio"
        )
        
    # Шаг 1: Сохраняем исходные фрагменты
    original_fragments = audio_fragments.copy()
    
    # Освобождаем память после каждого этапа
    import gc
    
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
    
    # Освобождаем память
    gc.collect()
    
    # Группа B1: Замедление записи в 0.8 раза (Пункт 2)
    # Применяем к результатам пункта 1
    slowdown_fragments = []
    speed_factor_slow = 0.8  # Только один фактор замедления
    
    for fragment in step1_fragments:
        try:
            slow_fragment = fast_change_speed(fragment, speed_factor_slow)
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
            fast_fragment = fast_change_speed(fragment, speed_factor_fast)
            # Проверяем длину после изменения скорости
            fast_fragment = ensure_min_length(fast_fragment)
            speedup_fragments.append(fast_fragment)
        except Exception as e:
            error_logger.log_error(f"Ошибка ускорения: {str(e)}", "augmentation", "augment_audio")
    
    # Объединяем результаты пунктов 2 и 3
    speed_modified_fragments = slowdown_fragments + speedup_fragments
    
    # Освобождаем память
    gc.collect()
    
    # Группа C: Временное маскирование (Пункт 4)
    # Применяем к результатам пунктов 2 и 3
    masked_fragments = []
    
    # Используем батчи для снижения расхода памяти
    for i in range(0, len(speed_modified_fragments), MAX_BATCH_SIZE):
        batch = speed_modified_fragments[i:i + MAX_BATCH_SIZE]
        for fragment in batch:
            try:
                masked_fragment = apply_time_masking(fragment)
                # Проверяем длину после маскирования
                masked_fragment = ensure_min_length(masked_fragment)
                masked_fragments.append(masked_fragment)
            except Exception as e:
                error_logger.log_error(f"Ошибка временного маскирования: {str(e)}", "augmentation", "augment_audio")
        
        # Освобождаем память после каждого батча
        gc.collect()
    
    # Объединяем все результаты:
    # 1. Исходные фрагменты (original_fragments)
    # 2. Фрагменты с удаленным шумом (denoised_fragments)
    # 3. Замедленные фрагменты из пунктов 1 и 2 (slowdown_fragments)
    # 4. Ускоренные фрагменты из пунктов 1 и 2 (speedup_fragments)
    # 5. Маскированные фрагменты из пунктов 3 и 4 (masked_fragments)
    result_fragments = original_fragments + denoised_fragments + slowdown_fragments + speedup_fragments + masked_fragments
    
    # Сборка мусора перед возвратом результата
    gc.collect()
    
    # Если результат превышает допустимый размер, случайно выбираем подмножество
    if len(result_fragments) > MAX_AUGMENTED_SAMPLES:
        import random
        result_fragments = random.sample(result_fragments, MAX_AUGMENTED_SAMPLES)
        error_logger.log_error(
            f"Результат аугментации превысил лимит и был ограничен до {MAX_AUGMENTED_SAMPLES} фрагментов",
            "augmentation", "augment_audio"
        )
    
    # Логирование результатов для отладки
    error_logger.log_error(
        f"Аугментация создала {len(result_fragments)} фрагментов из {len(audio_fragments)} исходных",
        "augmentation", "augment_audio"
    )
    
    return result_fragments

def remove_noise(audio_data):
    """
    Оптимизированная функция удаления шума для ускорения обработки
    """
    # Проверка на пустые данные
    if len(audio_data) == 0:
        return audio_data
    
    # Оптимизация: пропускаем обработку для коротких фрагментов
    if len(audio_data) < 2048:
        return audio_data
    
    try:
    # Расчет спектрограммы с использованием оптимизированных параметров
    n_fft = N_FFT
    
        # Предварительно проверяем размерность и подгоняем размер окна к длине аудио
        if n_fft > len(audio_data):
            n_fft = 2 ** int(np.log2(len(audio_data)))
            if n_fft < 512:  # слишком маленькое окно бесполезно для шумоподавления
                return audio_data
        
        # Корректный padding нужной длины
        padding_length = n_fft - (len(audio_data) % n_fft) if len(audio_data) % n_fft else 0
        padded_audio = np.pad(audio_data, (0, padding_length))
        
        # Используем более безопасный метод для расчета STFT
        # Создаем окно Ханна подходящего размера
        window = np.hanning(n_fft).astype(np.float32)
        
        # Делим сигнал на фреймы подходящего размера
        # и применяем быстрое преобразование Фурье
        hop = n_fft // 4
        frames = librosa.util.frame(padded_audio, frame_length=n_fft, hop_length=hop)
        frames = frames.T * window  # Применяем оконную функцию
        stft = np.fft.rfft(frames, axis=1)
    
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Адаптивное определение шума с оптимизированными параметрами
    noise_percentile = 15  # Нижний персентиль величин, вероятно, является шумом
    
        # Вычисление порога шума
    noise_thresh = np.percentile(mag, noise_percentile, axis=0)
    
    # Применяем мягкое спектральное вычитание с адаптивным порогом и векторизацией
    gain = 1.0 - (noise_thresh / (mag + 1e-10))
    gain = np.maximum(0.1, gain)  # Ограничиваем минимальное значение коэффициента усиления
    mag = mag * gain
    
    # Оптимизированное обратное преобразование
    stft_denoised = mag * np.exp(1j * phase)
        denoised_frames = np.fft.irfft(stft_denoised)
        
        # Восстанавливаем сигнал с overlap-add
        audio_denoised = np.zeros(len(padded_audio))
        window_sum = np.zeros(len(padded_audio))
        
        for i, frame in enumerate(denoised_frames):
            start = i * hop
            end = start + n_fft
            if end <= len(audio_denoised):
                audio_denoised[start:end] += frame * window
                window_sum[start:end] += window ** 2
        
        # Нормализуем по весу окон и обрезаем до исходной длины
        idx = window_sum > 1e-10
        audio_denoised[idx] /= window_sum[idx]
        
        return audio_denoised[:len(audio_data)]
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(
            f"Ошибка удаления шума: {str(e)}", 
            "audio", "remove_noise"
        )
        # Возвращаем исходные данные в случае ошибки
        return audio_data

def fast_change_speed(audio_data, speed_factor):
    """
    Оптимизированное изменение скорости аудиоданных
    с прямым ресемплированием, адаптировано для многопроцессорной обработки
    """
    try:
        # Проверка на пустые данные
        if len(audio_data) == 0:
            return audio_data
            
        # Проверка на минимальную длину аудио - оптимизация для коротких фрагментов
        if len(audio_data) < 1024:
            return audio_data
            
        # Проверка допустимости параметра
        if speed_factor <= 0:
            speed_factor = 1.0 
            
        # Ограничение длины обрабатываемого аудио
        max_length = 10 * SAMPLE_RATE  # 10 секунд максимум
        if len(audio_data) > max_length:
            audio_data = audio_data[:max_length]
            
        # Определяем необходимый размер выходного массива
        output_length = int(len(audio_data) / speed_factor)
        if output_length < 1024:  # Слишком короткий результат бесполезен
            return audio_data
            
        # Используем быстрое и простое ресемплирование
        indices = np.linspace(0, len(audio_data) - 1, output_length)
            indices = indices.astype(np.int32)
        result = audio_data[indices]
        return result
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(
            f"Ошибка изменения скорости: {str(e)}", 
            "audio", "fast_change_speed"
        )
        return audio_data

def fast_change_pitch(audio_data, n_steps, sr=16000):
    """
    Оптимизированное изменение высоты тона аудиоданных
    с использованием numpy и scipy, адаптировано для многопроцессорной обработки
    """
    try:
        # Проверка на пустые данные
        if len(audio_data) == 0 or np.all(audio_data == 0):
            return audio_data
            
        # Для очень коротких аудио пропускаем обработку
        if len(audio_data) < 1024:
            return audio_data
            
        # Ограничение длины обрабатываемого аудио
        max_length = 10 * sr  # 10 секунд максимум
        if len(audio_data) > max_length:
            audio_data = audio_data[:max_length]
            
        # Используем scipy.signal.resample вместо librosa.effects.pitch_shift
        # для более быстрого изменения высоты тона
        
        # Ограничение n_steps для предотвращения экстремальных значений
        n_steps = np.clip(n_steps, -4, 4)
        
        # Коэффициент изменения частоты дискретизации для изменения высоты тона
        rate = 2.0 ** (n_steps / 12.0)
        
        # Изменение высоты тона путем изменения частоты дискретизации с последующей компенсацией
        # 1. Ресемплирование аудио с новой частотой дискретизации
        new_length = int(len(audio_data) / rate)
        if new_length < 512:  # Слишком короткий результат бесполезен
            return audio_data
            
        # Используем scipy.signal.resample вместо scipy.signal.resample_poly для лучшей точности
        y_shifted = scipy.signal.resample(audio_data, new_length)
        
        # Если длина изменилась значительно, обрезаем или дополняем
        if len(y_shifted) > len(audio_data):
            # Обрезаем до исходной длины
            y_shifted = y_shifted[:len(audio_data)]
        elif len(y_shifted) < len(audio_data):
            # Дополняем нулями до исходной длины
            padding = np.zeros(len(audio_data) - len(y_shifted))
            y_shifted = np.concatenate((y_shifted, padding))
            
        return y_shifted
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(
            f"Ошибка изменения высоты тона: {str(e)}", 
            "audio", "fast_change_pitch"
        )
        # В случае ошибки возвращаем исходный сигнал
        return audio_data

def augment_audio_data(audio_fragments, labels, augmentation_factor=2):
    """
    Расширяет набор аудиоданных с помощью аугментации.
    Параллельно обрабатывает аудиофрагменты, применяя различные техники аугментации.
    
    Args:
        audio_fragments: список numpy массивов с аудиоданными
        labels: список соответствующих меток
        augmentation_factor: во сколько раз увеличить набор данных (по умолчанию 2)
        
    Returns:
        Tuple[List[np.ndarray], List[Any]]: кортеж (аугментированные аудиофрагменты, соответствующие метки)
    """
    try:
        if not audio_fragments or len(audio_fragments) == 0:
            error_logger.log_error("Ошибка: пустой список аудиофрагментов", "augmentation", "augment_audio_data")
            return audio_fragments, labels
            
        # Проверяем соответствие количества фрагментов и меток
        if len(audio_fragments) != len(labels):
            error_logger.log_error(
                f"Ошибка: количество аудиофрагментов ({len(audio_fragments)}) не соответствует количеству меток ({len(labels)})",
                "augmentation", "augment_audio_data"
            )
            return audio_fragments, labels
            
        start_count = len(audio_fragments)
        target_count = start_count * augmentation_factor
        augmented_fragments = audio_fragments.copy()
        augmented_labels = labels.copy()
        
        # Количество аудиофрагментов, которые нужно аугментировать
        fragments_to_augment = target_count - start_count
        
        # Если датасет слишком маленький, выполняем последовательную обработку
        if start_count < 4 or fragments_to_augment < 4 or N_JOBS <= 1:
            for _ in range(fragments_to_augment):
                # Выбираем случайный индекс из исходного набора
                original_idx = random.randint(0, start_count - 1)
                
                # Выбираем случайную технику аугментации
                augmentation_technique = random.choice([
                    'pitch_shift', 'time_stretch', 'add_noise', 'combined'
                ])
                
                # Применяем выбранную технику
                original_audio = audio_fragments[original_idx]
                augmented_audio = apply_augmentation(original_audio, augmentation_technique)
                
                if augmented_audio is not None:
                    augmented_fragments.append(augmented_audio)
                    augmented_labels.append(labels[original_idx])
        else:
            # Подготавливаем параметры для параллельной обработки
            augmentation_tasks = []
            for _ in range(fragments_to_augment):
                original_idx = random.randint(0, start_count - 1)
                augmentation_technique = random.choice([
                    'pitch_shift', 'time_stretch', 'add_noise', 'combined'
                ])
                augmentation_tasks.append((original_idx, audio_fragments[original_idx], augmentation_technique))
            
            # Используем ProcessPoolExecutor для параллельной обработки
            try:
                with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                    # Определяем вспомогательную функцию для параллельной обработки
                    def process_augmentation(task):
                        idx, audio, technique = task
                        try:
                            augmented = apply_augmentation(audio, technique)
                            return (idx, augmented)
                        except Exception as e:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                            line_no = exc_tb.tb_lineno
                            print(f"{fname} - {line_no} - {str(e)}")
                            
                            error_logger.log_error(
                                f"Ошибка аугментации фрагмента: {str(e)}", 
                                "augmentation", "process_augmentation"
                            )
                            return (idx, None)
                    
                    # Запускаем параллельную обработку
                    results = list(executor.map(process_augmentation, augmentation_tasks))
                    
                    # Обрабатываем результаты
                    for idx, augmented_audio in results:
                        if augmented_audio is not None:
                            augmented_fragments.append(augmented_audio)
                            augmented_labels.append(labels[idx])
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                
                error_logger.log_error(
                    f"Ошибка параллельной аугментации: {str(e)}", 
                    "augmentation", "augment_audio_data"
                )
                # В случае ошибки возвращаем текущий (возможно частично аугментированный) набор
                
        final_count = len(augmented_fragments)
        if final_count < target_count:
            error_logger.log_error(
                f"Предупреждение: итоговый размер набора данных ({final_count}) меньше ожидаемого ({target_count})",
                "augmentation", "augment_audio_data"
            )
            
        return augmented_fragments, augmented_labels
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(f"Ошибка аугментации аудиоданных: {str(e)}", "augmentation", "augment_audio_data")
        # В случае ошибки возвращаем исходный набор
        return audio_fragments, labels

def apply_augmentation(audio, technique):
    """
    Применяет выбранную технику аугментации к аудиофрагменту.
    
    Args:
        audio: numpy массив с аудиоданными
        technique: строка с названием техники ('pitch_shift', 'time_stretch', 'add_noise', 'combined')
        
    Returns:
        np.ndarray: аугментированный аудиофрагмент или None в случае ошибки
    """
    try:
        if audio is None or len(audio) == 0:
            return None
            
        if technique == 'pitch_shift':
            # Изменение высоты тона
            pitch_shift = random.uniform(-MAX_PITCH_SHIFT, MAX_PITCH_SHIFT)
            return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=pitch_shift)
            
        elif technique == 'time_stretch':
            # Изменение скорости воспроизведения
            speed_factor = random.uniform(MIN_SPEED, MAX_SPEED)
            return librosa.effects.time_stretch(audio, rate=speed_factor)
            
        elif technique == 'add_noise':
            # Добавление шума с контролируемым SNR
            return add_noise_with_controlled_snr(audio)
            
        elif technique == 'combined':
            # Комбинация нескольких техник
            # Сначала меняем скорость
            speed_factor = random.uniform(MIN_SPEED, MAX_SPEED)
            audio_aug = librosa.effects.time_stretch(audio, rate=speed_factor)
            
            # Затем добавляем небольшой сдвиг тона
            pitch_shift = random.uniform(-MAX_PITCH_SHIFT/2, MAX_PITCH_SHIFT/2)
            audio_aug = librosa.effects.pitch_shift(audio_aug, sr=SAMPLE_RATE, n_steps=pitch_shift)
            
            # И наконец добавляем немного шума
            snr_db = random.uniform(MAX_SNR_DB, MAX_SNR_DB + 5)  # Немного больше SNR для меньшего шума
            return add_noise_with_snr(audio_aug, snr_db)
            
        else:
            error_logger.log_error(f"Неизвестная техника аугментации: {technique}", "augmentation", "apply_augmentation")
            return audio  # Возвращаем исходный аудиофрагмент без изменений
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(f"Ошибка при применении аугментации: {str(e)}", "augmentation", "apply_augmentation")
        return None

def add_noise_with_controlled_snr(audio):
    """
    Добавляет шум к аудиосигналу с контролируемым отношением сигнал/шум.
    
    Args:
        audio: numpy массив с аудиоданными
        
    Returns:
        np.ndarray: аудиоданные с добавленным шумом
    """
    try:
        # Выбираем случайное значение SNR из диапазона
        snr_db = random.uniform(MIN_SNR_DB, MAX_SNR_DB)
        return add_noise_with_snr(audio, snr_db)
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(f"Ошибка при добавлении шума: {str(e)}", "augmentation", "add_noise_with_controlled_snr")
        return audio

def add_noise_with_snr(audio, snr_db):
    """
    Добавляет белый шум к аудиосигналу с заданным отношением сигнал/шум в дБ.
    
    Args:
        audio: numpy массив с аудиоданными
        snr_db: отношение сигнал/шум в дБ
        
    Returns:
        np.ndarray: аудиоданные с добавленным шумом
    """
    try:
        # Преобразуем SNR из дБ в линейную шкалу
        snr_linear = 10 ** (snr_db / 10)
        
        # Расчет среднеквадратичной мощности сигнала
        audio_rms = np.sqrt(np.mean(audio ** 2))
        
        # Расчет необходимой мощности шума
        noise_rms = audio_rms / snr_linear
        
        # Генерация белого шума
        noise = np.random.normal(0, noise_rms, len(audio))
        
        # Добавление шума к сигналу
        noisy_audio = audio + noise
        
        # Нормализация для предотвращения клиппинга
        max_amplitude = np.max(np.abs(noisy_audio))
        if max_amplitude > 1.0:
            noisy_audio = noisy_audio / max_amplitude
            
        return noisy_audio
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(f"Ошибка при добавлении шума с SNR: {str(e)}", "augmentation", "add_noise_with_snr")
        return audio

def batch_augment(audio_fragments, labels, augmentation_techniques, parallel=True):
    """
    Выполняет пакетную аугментацию аудиоданных с применением набора техник.
    Поддерживает параллельную обработку для больших наборов данных.
    
    Args:
        audio_fragments: список numpy массивов с аудиоданными
        labels: список соответствующих меток
        augmentation_techniques: список техник аугментации для применения
        parallel: использовать ли параллельную обработку
        
    Returns:
        Tuple[List[np.ndarray], List[Any]]: кортеж (аугментированные аудиофрагменты, соответствующие метки)
    """
    try:
        if not audio_fragments or len(audio_fragments) == 0:
            return audio_fragments, labels
            
        if not augmentation_techniques or len(augmentation_techniques) == 0:
            return audio_fragments, labels
            
        augmented_fragments = []
        augmented_labels = []
        
        # Для каждой техники создаем копии всех аудиофрагментов с этой аугментацией
        for technique in augmentation_techniques:
            # Если можно выполнить параллельную обработку и у нас достаточно данных
            if parallel and len(audio_fragments) >= 4 and N_JOBS > 1:
                try:
                    # Создаем частичную функцию для конкретной техники
                    aug_func = partial(apply_augmentation, technique=technique)
                    
                    # Параллельно применяем аугментацию ко всем фрагментам
                    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                        augmented_batch = list(executor.map(aug_func, audio_fragments))
                        
                    # Фильтруем None результаты и добавляем в общий список
                    for i, augmented in enumerate(augmented_batch):
                        if augmented is not None:
                            augmented_fragments.append(augmented)
                            augmented_labels.append(labels[i])
                            
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                    line_no = exc_tb.tb_lineno
                    print(f"{fname} - {line_no} - {str(e)}")
                    
                    error_logger.log_error(
                        f"Ошибка при параллельной пакетной аугментации: {str(e)}", 
                        "augmentation", "batch_augment"
                    )
                    # В случае ошибки выполняем последовательную обработку
                    for i, audio in enumerate(audio_fragments):
                        try:
                            augmented = apply_augmentation(audio, technique)
                            if augmented is not None:
                                augmented_fragments.append(augmented)
                                augmented_labels.append(labels[i])
                        except Exception as e2:
                            error_logger.log_error(
                                f"Ошибка при обработке фрагмента {i}: {str(e2)}", 
                                "augmentation", "batch_augment"
                            )
                            continue
            else:
                # Последовательная обработка
                for i, audio in enumerate(audio_fragments):
                    try:
                        augmented = apply_augmentation(audio, technique)
                        if augmented is not None:
                            augmented_fragments.append(augmented)
                            augmented_labels.append(labels[i])
                    except Exception as e:
                        error_logger.log_error(
                            f"Ошибка при обработке фрагмента {i} с техникой {technique}: {str(e)}", 
                            "augmentation", "batch_augment"
                        )
                        continue
        
        # Объединяем исходный и аугментированный наборы
        all_fragments = audio_fragments + augmented_fragments
        all_labels = labels + augmented_labels
        
        return all_fragments, all_labels
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(f"Ошибка при пакетной аугментации: {str(e)}", "augmentation", "batch_augment")
        return audio_fragments, labels
