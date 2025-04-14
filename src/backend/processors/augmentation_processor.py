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

# Константы для оптимизации
MAX_AUGMENTED_SAMPLES = 50  # Максимальное количество аугментированных образцов
N_FFT = 2048
HOP_LENGTH = 512

# Определяем оптимальное количество процессов
# Оставляем 1 ядро для основных операций системы
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# Константы для работы с аудио и аугментации
SAMPLE_RATE = 16000  # Стандартная частота дискретизации
MIN_SNR_DB = 5       # Минимальное отношение сигнал/шум для добавления шума
MAX_SNR_DB = 15      # Максимальное отношение сигнал/шум для добавления шума
MAX_PITCH_SHIFT = 2  # Максимальное изменение высоты тона (в полутонах)
MIN_SPEED = 0.9      # Минимальный коэффициент изменения скорости
MAX_SPEED = 1.1      # Максимальный коэффициент изменения скорости

def augment_audio(audio_fragments):
    """
    Оптимизированная аугментация аудиофрагментов с параллельной обработкой
    Возвращает расширенный набор аудиофрагментов с ограничением
    на максимальное количество сэмплов для повышения производительности
    """
    if not audio_fragments:
        return []
        
    # Оптимизация: ограничиваем количество исходных фрагментов
    if len(audio_fragments) > 5:
        # Если фрагментов много, выбираем случайные 5 для аугментации
        audio_fragments = random.sample(audio_fragments, 5)
    
    # Шаг 1: Исходные фрагменты
    result_fragments = []
    result_fragments.extend(audio_fragments)
    
    # Шаг 2: Удаление шума (группа A)
    # Параллельное удаление шума, если достаточно фрагментов и процессоров
    sample_for_denoising = audio_fragments[:min(3, len(audio_fragments))]
    
    if len(sample_for_denoising) > 1 and N_JOBS > 1:
        try:
            with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                denoised_fragments = list(executor.map(remove_noise, sample_for_denoising))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            
            error_logger.log_error(f"Параллельное удаление шума не удалось: {str(e)}", "processing", "augmentation")
            denoised_fragments = [remove_noise(fragment) for fragment in sample_for_denoising]
    else:
        denoised_fragments = [remove_noise(fragment) for fragment in sample_for_denoising]
        
    result_fragments.extend(denoised_fragments)
    
    # Оптимизация: ограничиваем количество аугментаций при большом объеме данных
    if len(audio_fragments) >= 3:
        # Для большого набора используем меньшее количество скоростей
        speeds = [0.8, 1.2]
    else:
        # Для малого набора используем полный набор скоростей
        speeds = [0.8, 0.9, 1.1, 1.2]
    
    # Набор после шагов 1-2
    step1_2_fragments = result_fragments.copy()
    
    # Оптимизация: ограничиваем базовый набор для дальнейшей аугментации
    if len(step1_2_fragments) > 8:
        step1_2_fragments = random.sample(step1_2_fragments, 8)
    
    # Шаг 3-4: Параллельное изменение скорости
    all_speed_tasks = []
    for fragment in step1_2_fragments:
        for speed in speeds:
            all_speed_tasks.append((fragment, speed))
    
    # Параллельная обработка для изменения скорости, если задач достаточно
    if len(all_speed_tasks) >= 4 and N_JOBS > 1:
        try:
            with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                # Распаковываем аргументы для параллельного выполнения
                speed_fragments = list(executor.map(
                    lambda args: fast_change_speed(*args),
                    all_speed_tasks
                ))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            
            error_logger.log_error(f"Параллельное изменение скорости не удалось: {str(e)}", "processing", "augmentation")
            speed_fragments = [fast_change_speed(fragment, speed) for fragment, speed in all_speed_tasks]
    else:
        speed_fragments = [fast_change_speed(fragment, speed) for fragment, speed in all_speed_tasks]
    
    result_fragments.extend(speed_fragments)
    
    # Оптимизация: проверяем, не превышаем ли лимит
    if len(result_fragments) > MAX_AUGMENTED_SAMPLES:
        # Если превышаем, случайно выбираем подмножество
        result_fragments = random.sample(result_fragments, MAX_AUGMENTED_SAMPLES)
        return result_fragments
    
    # Оптимизация: ограничиваем базовый набор для тональных изменений
    steps_1_4_fragments = result_fragments.copy()
    if len(steps_1_4_fragments) > 10:
        steps_1_4_fragments = random.sample(steps_1_4_fragments, 10)
    
    # Шаг 5: Параллельное изменение тональности
    pitch_shifts = [-2, 2]  # Только крайние значения
    samples_for_pitch = steps_1_4_fragments[:min(5, len(steps_1_4_fragments))]
    
    # Создаем список задач для параллельной обработки
    all_pitch_tasks = []
    for fragment in samples_for_pitch:
        for pitch_shift in pitch_shifts:
            all_pitch_tasks.append((fragment, pitch_shift))
    
    # Параллельная обработка для изменения тональности, если задач достаточно
    if len(all_pitch_tasks) >= 4 and N_JOBS > 1:
        try:
            with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                # Распаковываем аргументы для параллельного выполнения
                pitch_fragments = list(executor.map(
                    lambda args: fast_change_pitch(*args),
                    all_pitch_tasks
                ))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            
            error_logger.log_error(f"Параллельное изменение тональности не удалось: {str(e)}", "processing", "augmentation")
            pitch_fragments = [fast_change_pitch(fragment, pitch_shift) for fragment, pitch_shift in all_pitch_tasks]
    else:
        pitch_fragments = [fast_change_pitch(fragment, pitch_shift) for fragment, pitch_shift in all_pitch_tasks]
    
    result_fragments.extend(pitch_fragments)
    
    # Оптимизация: ограничиваем итоговое количество фрагментов
    if len(result_fragments) > MAX_AUGMENTED_SAMPLES:
        # Если итоговый набор слишком большой, ограничиваем его
        result_fragments = random.sample(result_fragments, MAX_AUGMENTED_SAMPLES)
    
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
    
    # Расчет спектрограммы с использованием оптимизированных параметров
    n_fft = N_FFT
    
    # Быстрое вычисление спектрограммы напрямую через numpy
    # вместо librosa.stft для ускорения
    stft = np.fft.rfft(np.hanning(n_fft).astype(np.float32) * 
                      np.pad(audio_data, (0, n_fft - len(audio_data) % n_fft if len(audio_data) % n_fft else 0)))
    
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Адаптивное определение шума с оптимизированными параметрами
    noise_percentile = 15  # Нижний персентиль величин, вероятно, является шумом
    
    # Оптимизация: вычисление порога шума
    # Используем axis=0 вместо axis=1 из-за другой формы массива от np.fft.rfft
    noise_thresh = np.percentile(mag, noise_percentile, axis=0)
    noise_thresh = noise_thresh[:, np.newaxis]
    
    # Применяем мягкое спектральное вычитание с адаптивным порогом и векторизацией
    gain = 1.0 - (noise_thresh / (mag + 1e-10))
    gain = np.maximum(0.1, gain)  # Ограничиваем минимальное значение коэффициента усиления
    mag = mag * gain
    
    # Оптимизированное обратное преобразование
    stft_denoised = mag * np.exp(1j * phase)
    audio_denoised = np.fft.irfft(stft_denoised)[:len(audio_data)]
    
    return audio_denoised

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
            
        # Оптимизация: используем более эффективный метод ресемплирования
        # для изменения скорости вместо librosa.effects.time_stretch
        # Это быстрее для нашего случая
        
        # Для ускорения фрагмента уменьшаем его длину
        if speed_factor > 1.0:
            # Рассчитываем новую длину
            new_length = int(len(audio_data) / speed_factor)
            # Быстрое ресемплирование с использованием линейной интерполяции
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            indices = indices.astype(np.int32)
            # Выборка значений по индексам
            y_stretch = audio_data[indices]
        # Для замедления фрагмента увеличиваем его длину
        else:
            # Рассчитываем новую длину
            new_length = int(len(audio_data) * 1/speed_factor)
            # Быстрое ресемплирование с использованием линейной интерполяции
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            # Интерполируем значения
            y_stretch = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        # Обеспечиваем, что результат имеет ту же длину, что и вход
        target_length = len(audio_data)
        if len(y_stretch) > target_length:
            y_stretch = y_stretch[:target_length]
        elif len(y_stretch) < target_length:
            # Дополняем нулями вместо отражения для скорости
            padding = np.zeros(target_length - len(y_stretch), dtype=audio_data.dtype)
            y_stretch = np.concatenate([y_stretch, padding])
            
        return y_stretch
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        # В многопроцессорном режиме логирование ошибок может не работать корректно
        # error_message = f"Ошибка при изменении скорости: {str(e)}"
        # error_logger.log_error(error_message, "processing", "augmentation")
        return audio_data

def fast_change_pitch(audio_data, n_steps, sr=16000):
    """
    Оптимизированное изменение высоты тона без сохранения длительности
    Использует более быстрый подход с прямым ресемплированием
    Адаптировано для многопроцессорной обработки
    """
    try:
        # Проверка на пустые данные
        if len(audio_data) == 0:
            return audio_data
            
        # Проверка на минимальную длину аудио
        if len(audio_data) < 1024:
            return audio_data
        
        # Оптимизация: используем прямой метод изменения высоты тона
        # через ресемплирование без использования librosa.effects.pitch_shift
        # Это значительно быстрее для наших целей
        
        # Рассчитываем коэффициент ресемплирования
        # ~5.9% на полутон
        rate = 2.0 ** (n_steps / 12.0)
        
        # Ресемплируем сигнал для изменения высоты тона
        # Для более высокого тона уменьшаем длину, а затем восстанавливаем исходную длину
        # Для более низкого тона увеличиваем длину, а затем обрезаем до исходной длины
        
        # Шаг 1: Ресемплирование
        if rate > 1.0:  # Повышение тона
            # Уменьшаем длину
            indices = np.round(np.linspace(0, len(audio_data) - 1, int(len(audio_data) / rate)))
            y_shifted = audio_data[indices.astype(int)]
        else:  # Понижение тона
            # Увеличиваем длину
            indices = np.linspace(0, len(audio_data) - 1, int(len(audio_data) * 1/rate))
            y_shifted = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        # Шаг 2: Восстановление исходной длины
        target_length = len(audio_data)
        if len(y_shifted) > target_length:
            y_shifted = y_shifted[:target_length]
        elif len(y_shifted) < target_length:
            padding = np.zeros(target_length - len(y_shifted), dtype=audio_data.dtype)
            y_shifted = np.concatenate([y_shifted, padding])
            
        return y_shifted
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        # В многопроцессорном режиме логирование ошибок может не работать корректно
        # error_message = f"Ошибка при изменении высоты тона: {str(e)}"
        # error_logger.log_error(error_message, "processing", "augmentation")
        return audio_data  # В случае ошибки возвращаем оригинальные данные

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
