import numpy as np
import librosa
import random
from backend.api.error_logger import error_logger
from backend.config import (
    AUGMENTATION_PROCESSOR, SAMPLE_RATE, HOP_LENGTH, AUDIO_FRAGMENT_LENGTH
)
import sys
import os
import scipy.signal

# Используем константы из конфигурационного файла
MIN_SNR_DB = AUGMENTATION_PROCESSOR['MIN_SNR_DB']
MAX_SNR_DB = AUGMENTATION_PROCESSOR['MAX_SNR_DB']
MAX_PITCH_SHIFT = AUGMENTATION_PROCESSOR['MAX_PITCH_SHIFT']
MIN_SPEED = AUGMENTATION_PROCESSOR['MIN_SPEED']
MAX_SPEED = AUGMENTATION_PROCESSOR['MAX_SPEED']

# Параметры временного маскирования
MASK_COUNT_MIN = AUGMENTATION_PROCESSOR['MASK_COUNT_MIN']
MASK_COUNT_MAX = AUGMENTATION_PROCESSOR['MASK_COUNT_MAX']
MASK_LENGTH_MIN = AUGMENTATION_PROCESSOR['MASK_LENGTH_MIN']
MASK_LENGTH_MAX = AUGMENTATION_PROCESSOR['MASK_LENGTH_MAX']
N_FFT_AUGMENTATION = AUGMENTATION_PROCESSOR['N_FFT_AUGMENTATION']


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
        return np.zeros(AUDIO_FRAGMENT_LENGTH)  # Возвращаем тишину минимальной длины
        
    if len(audio_fragment) < AUDIO_FRAGMENT_LENGTH:
        # Используем циклическое повторение (вместо простого дополнения нулями)
        # для сохранения характеристик сигнала
        multiplier = int(np.ceil(AUDIO_FRAGMENT_LENGTH / len(audio_fragment)))
        extended_fragment = np.tile(audio_fragment, multiplier)[:AUDIO_FRAGMENT_LENGTH]
        
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
    
    # Пункт 2: К набору из пункта 1 применяем маскирование к результатам изменения скорости
    # Применяем к slowdown_fragments
    result_fragments = []
    
    for fragment in slowdown_fragments:
        try:
            # Добавляем исходный
            result_fragments.append(fragment)
            
            # Добавляем с временным маскированием
            masked_fragment = apply_time_masking(fragment)
            # Проверяем длину после маскирования
            masked_fragment = ensure_min_length(masked_fragment)
            result_fragments.append(masked_fragment)
        except Exception as e:
            error_logger.log_error(f"Ошибка маскирования: {str(e)}", "augmentation", "augment_audio")
    
    # Пункт 3: К набору из пункта 1 применяем маскирование к результатам изменения скорости
    # Применяем к speedup_fragments
    for fragment in speedup_fragments:
        try:
            # Добавляем исходный
            result_fragments.append(fragment)
            
            # Добавляем с временным маскированием
            masked_fragment = apply_time_masking(fragment)
            # Проверяем длину после маскирования
            masked_fragment = ensure_min_length(masked_fragment)
            result_fragments.append(masked_fragment)
        except Exception as e:
            error_logger.log_error(f"Ошибка маскирования: {str(e)}", "augmentation", "augment_audio")
        
    return result_fragments

def remove_noise(audio_data):
    """
    Удаление шума с использованием спектрального маскирования
    и минимально вычислительного подхода
    """
    # Проверка на пустые данные
    if audio_data is None or len(audio_data) == 0:
        return audio_data
        
    # Для очень коротких аудио пропускаем шумоподавление
    if len(audio_data) < 2048:
        return audio_data
        
    try:
        # Расчет спектрограммы с использованием оптимизированных параметров
        n_fft = N_FFT_AUGMENTATION
        
        # Предварительно проверяем размерность и подгоняем размер окна к длине аудио
        if len(audio_data) < n_fft:
            # Для очень коротких аудио используем меньший размер окна
            n_fft = 512  # Если аудио короче, используем меньшее окно
            
        # Получаем спектрограмму мощности
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=HOP_LENGTH)
        magnitude = np.abs(stft)
        power = magnitude ** 2
        
        # Адаптивное определение шума с оптимизированными параметрами
        noise_percentile = 15  # Нижний персентиль величин, вероятно, является шумом
        
        # Вычисление порога шума
        noise_thresh = np.percentile(magnitude, noise_percentile, axis=0)
        
        # Применяем мягкое спектральное вычитание с адаптивным порогом и векторизацией
        gain = 1.0 - (noise_thresh / (magnitude + 1e-10))
        gain = np.maximum(0.1, gain)  # Ограничиваем минимальное значение коэффициента усиления
        mag = magnitude * gain
        
        # Оптимизированное обратное преобразование
        stft_denoised = mag * np.exp(1j * np.angle(stft))
        denoised_frames = np.fft.irfft(stft_denoised)
        
        # Восстанавливаем сигнал с overlap-add
        audio_denoised = np.zeros(len(audio_data))
        window_sum = np.zeros(len(audio_data))
        
        for i, frame in enumerate(denoised_frames):
            start = i * (n_fft // 4)
            end = start + n_fft
            if end <= len(audio_denoised):
                audio_denoised[start:end] += frame
                window_sum[start:end] += 1
        
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

def fast_change_pitch(audio_data, n_steps, sr=SAMPLE_RATE):
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
    Аугментирует аудиофрагменты, создавая несколько версий с различными преобразованиями
    
    Args:
        audio_fragments: список аудиофрагментов (numpy arrays)
        labels: список соответствующих меток
        augmentation_factor: во сколько раз увеличить датасет
        
    Returns:
        tuple (augmented_fragments, augmented_labels) с аугментированными данными
    """
    try:
        if not audio_fragments or not labels:
            return [], []
            
        # Проверяем наличие данных и соответствие размеров
        if len(audio_fragments) != len(labels):
            error_logger.log_error(
                f"Несоответствие размеров: {len(audio_fragments)} фрагментов и {len(labels)} меток",
                "augmentation", "augment_audio_data"
            )
            return audio_fragments, labels
            
        start_count = len(audio_fragments)
        
        # Если слишком мало фрагментов, просто дублируем их
        if start_count < 4:
            result_fragments = []
            result_labels = []
            
            for i in range(min(12, augmentation_factor)):
                for frag, label in zip(audio_fragments, labels):
                    result_fragments.append(frag.copy())
                    result_labels.append(label)
                    
            error_logger.log_error(
                f"Слишком мало исходных фрагментов ({start_count}), просто дублируем их",
                "augmentation", "augment_audio_data"
            )
            return result_fragments, result_labels
           
        # Готовим данные для аугментации
        fragments_by_label = {}
        
        for frag, label in zip(audio_fragments, labels):
            if label not in fragments_by_label:
                fragments_by_label[label] = []
            fragments_by_label[label].append(frag)
            
        # Аугментируем фрагменты по каждой метке
        augmented_fragments = []
        augmented_labels = []
        fragments_to_augment = sum(len(frags) for frags in fragments_by_label.values())
        
        # Применяем аугментацию последовательно
        for label, frags in fragments_by_label.items():
            # Применяем аугментацию напрямую
            augmented = augment_audio(frags)
            
            # Добавляем результаты
            augmented_fragments.extend(augmented)
            augmented_labels.extend([label] * len(augmented))
                
            # Добавляем оригинальные фрагменты
            augmented_fragments.extend(frags)
            augmented_labels.extend([label] * len(frags))
                
        # Проверяем результаты
        if not augmented_fragments:
            error_logger.log_error(
                "Аугментация не дала результатов, возвращаем исходные данные",
                "augmentation", "augment_audio_data"
            )
            return audio_fragments, labels
            
        # Логгируем
        error_logger.log_error(
            f"Аугментация завершена: {start_count} -> {len(augmented_fragments)} фрагментов",
            "augmentation", "augment_audio_data"
        )
            
        return augmented_fragments, augmented_labels
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
            
        error_logger.log_error(
            f"Ошибка аугментации: {str(e)}",
            "augmentation", "augment_audio_data"
        )
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

def batch_augment(audio_fragments, labels, augmentation_techniques):
    """
    Применяет список техник аугментации к аудиофрагментам в пакетном режиме
    
    Args:
        audio_fragments: список аудиофрагментов (numpy arrays)
        labels: список соответствующих меток
        augmentation_techniques: список функций аугментации
        
    Returns:
        tuple (augmented_fragments, augmented_labels) с аугментированными данными
    """
    try:
        # Проверяем на корректность входных данных
        if not audio_fragments or not labels:
            return [], []
            
        if len(audio_fragments) != len(labels):
            error_logger.log_error(
                "Количество аудиофрагментов и меток не совпадает",
                "augmentation", "batch_augment"
            )
            return audio_fragments, labels
            
        # Теперь выполняем последовательно для всех фрагментов
        result_fragments = []
        result_labels = []
            
        # Добавляем исходные фрагменты
        result_fragments.extend(audio_fragments)
        result_labels.extend(labels)
            
        # Применяем каждую технику аугментации
        for technique in augmentation_techniques:
            technique_name = technique.__name__ if hasattr(technique, "__name__") else "unknown_technique"
                
            try:
                for i, (fragment, label) in enumerate(zip(audio_fragments, labels)):
                    # Применяем технику аугментации
                    augmented = apply_augmentation(fragment, technique)
                        
                    # Проверяем результат
                    if augmented is not None and len(augmented) > 0:
                        # Добавляем новый фрагмент
                        result_fragments.append(augmented)
                        result_labels.append(label)
            except Exception as e:
                error_logger.log_error(
                    f"Ошибка при применении техники {technique_name}: {str(e)}",
                    "augmentation", "batch_augment"
                )
                    
        # Логгируем результаты
        error_logger.log_error(
            f"Пакетная аугментация завершена: {len(audio_fragments)} -> {len(result_fragments)} фрагментов",
            "augmentation", "batch_augment"
        )
            
        return result_fragments, result_labels
            
    except Exception as e:
        error_logger.log_error(f"Ошибка при пакетной аугментации: {str(e)}", "augmentation", "batch_augment")
        return audio_fragments, labels
