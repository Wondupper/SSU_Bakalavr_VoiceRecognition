import numpy as np
import librosa
import scipy.signal
from functools import lru_cache  # Добавлен импорт для кэширования
from concurrent.futures import ProcessPoolExecutor  # Добавляем поддержку многопроцессорности
import multiprocessing
from .augmentation_processor import augment_audio
from backend.api.error_logger import error_logger
import sys
import os
import random

# Константы для вычисления признаков
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
PREEMPHASIS_COEF = 0.97
MAX_FEATURE_LENGTH = 3  # Уменьшено с 5 до 3 секунд для снижения потребления памяти
SAMPLE_RATE = 16000     # Частота дискретизации
# Максимальное количество фреймов в выходном признаке (для стандартизации выхода)
# ~3 секунды при стандартных настройках
MAX_FRAMES = 100

# Определяем оптимальное количество процессов
# Ограничиваем максимальное количество процессоров для снижения нагрузки на память
N_JOBS = max(1, min(2, multiprocessing.cpu_count() - 1))  # Не более 2 процессов
# Ограничение для мини-батчей при обработке наборов данных
MAX_BATCH_SIZE = 8  # Ограничиваем размер батча для снижения потребления памяти

def create_voice_id_dataset(audio_fragments, name):
    """
    Создание датасета для модели идентификации по голосу
    с параллельной обработкой фрагментов
    """
    if not audio_fragments:
        raise ValueError("Нет аудиофрагментов для создания датасета")
    
    # УМЕНЬШЕНО: Ограничиваем количество фрагментов для экономии памяти
    MAX_FRAGMENTS = 3  # Уменьшено с 10 до 3
    if len(audio_fragments) > MAX_FRAGMENTS:
        # Берем случайные фрагменты
        audio_fragments = random.sample(audio_fragments, MAX_FRAGMENTS)
        error_logger.log_error(
            f"Количество фрагментов ограничено до {MAX_FRAGMENTS} для экономии памяти", 
            "processing", "dataset_creator"
        )
    
    # ИЗМЕНЕНО: Принудительная сборка мусора перед аугментацией
    import gc
    gc.collect()
    
    # Аугментация аудиофрагментов с ограничением результатов
    try:
        augmented_fragments = augment_audio(audio_fragments)
    except Exception as e:
        error_logger.log_error(
            f"Ошибка при аугментации: {str(e)}. Продолжаем без аугментации.", 
            "processing", "dataset_creator"
        )
        # Если аугментация не удалась, используем только исходные фрагменты
        augmented_fragments = audio_fragments.copy()
    
    # УМЕНЬШЕНО: Дополнительно ограничиваем количество аугментированных фрагментов
    MAX_AUGMENTED = MAX_BATCH_SIZE * 2  # Уменьшено в 2.5 раза
    if len(augmented_fragments) > MAX_AUGMENTED:
        augmented_fragments = random.sample(augmented_fragments, MAX_AUGMENTED)
        error_logger.log_error(
            f"Количество аугментированных фрагментов ограничено до {MAX_AUGMENTED}", 
            "processing", "dataset_creator"
        )
    
    # Сбор мусора после аугментации
    gc.collect()
    
    # Создание датасета для обучения нейронной сети
    # Используем пакетную обработку вместо обработки всего сразу
    dataset = []
    
    # Разбиваем на батчи для экономии памяти
    batch_size = min(MAX_BATCH_SIZE, len(augmented_fragments))
    
    for i in range(0, len(augmented_fragments), batch_size):
        batch = augmented_fragments[i:i+batch_size]
        
        # ИЗМЕНЕНО: Предпочитаем последовательную обработку для стабильности
        sequential_processing = True
        
        # Параллельная обработка батча если возможно и размер достаточно большой
        if len(batch) >= 4 and N_JOBS > 1 and not sequential_processing:
            try:
                with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                    features_list = list(executor.map(extract_features, batch))
                
                # Добавляем обработанный батч в датасет
                for features in features_list:
                    dataset.append({'features': features, 'label': name})
            except Exception as e:
                # Последовательное извлечение признаков при ошибке
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                
                error_logger.log_error(f"Параллельная обработка не удалась: {str(e)}, переключаемся на последовательную", 
                                      "processing", "dataset_creator")
                
                for fragment in batch:
                    try:
                        features = extract_features(fragment)
                        dataset.append({'features': features, 'label': name})
                    except Exception as inner_e:
                        error_logger.log_error(f"Ошибка при извлечении признаков: {str(inner_e)}", 
                                            "processing", "dataset_creator")
        else:
            # Последовательное извлечение признаков
            for fragment in batch:
                try:
                    features = extract_features(fragment)
                    dataset.append({'features': features, 'label': name})
                except Exception as e:
                    error_logger.log_error(f"Ошибка при извлечении признаков: {str(e)}", 
                                        "processing", "dataset_creator")
        
        # Принудительная сборка мусора после обработки каждого батча
        gc.collect()
    
    # Проверка наличия данных в датасете
    if not dataset:
        raise ValueError("Не удалось создать датасет: все операции извлечения признаков завершились с ошибками")
    
    return dataset

def create_emotion_dataset(audio_fragments, emotion):
    """
    Создание датасета для модели распознавания эмоций
    с параллельной обработкой фрагментов
    
    Args:
        audio_fragments: Список аудиофрагментов для обучения
        emotion: Строка с названием эмоции ('гнев', 'радость', 'грусть')
    
    Returns:
        dataset: Список словарей с признаками и метками для обучения модели
    """
    if not audio_fragments:
        raise ValueError("Нет аудиофрагментов для создания датасета")
    
    if emotion not in ['гнев', 'радость', 'грусть']:
        raise ValueError("Недопустимая эмоция. Допустимые: гнев, радость, грусть")
    
    # УМЕНЬШЕНО: Ограничиваем количество фрагментов для экономии памяти
    MAX_FRAGMENTS = 3  # Уменьшено с 10 до 3
    if len(audio_fragments) > MAX_FRAGMENTS:
        # Берем случайные фрагменты
        audio_fragments = random.sample(audio_fragments, MAX_FRAGMENTS)
        error_logger.log_error(
            f"Количество фрагментов ограничено до {MAX_FRAGMENTS} для экономии памяти", 
            "processing", "dataset_creator"
        )
    
    # ИЗМЕНЕНО: Принудительная сборка мусора перед аугментацией
    import gc
    gc.collect()
    
    # Аугментация аудиофрагментов с ограничением результатов
    try:
        augmented_fragments = augment_audio(audio_fragments)
    except Exception as e:
        error_logger.log_error(
            f"Ошибка при аугментации: {str(e)}. Продолжаем без аугментации.", 
            "processing", "dataset_creator"
        )
        # Если аугментация не удалась, используем только исходные фрагменты
        augmented_fragments = audio_fragments.copy()
    
    # УМЕНЬШЕНО: Дополнительно ограничиваем количество аугментированных фрагментов
    MAX_AUGMENTED = MAX_BATCH_SIZE * 2  # Уменьшено в 2.5 раза
    if len(augmented_fragments) > MAX_AUGMENTED:
        augmented_fragments = random.sample(augmented_fragments, MAX_AUGMENTED)
        error_logger.log_error(
            f"Количество аугментированных фрагментов ограничено до {MAX_AUGMENTED}", 
            "processing", "dataset_creator"
        )
    
    # Сбор мусора после аугментации
    gc.collect()
    
    # Создание датасета для обучения нейронной сети
    dataset = []
    
    # Разбиваем на батчи для экономии памяти
    batch_size = min(MAX_BATCH_SIZE, len(augmented_fragments))
    
    for i in range(0, len(augmented_fragments), batch_size):
        batch = augmented_fragments[i:i+batch_size]
        
        # ИЗМЕНЕНО: Предпочитаем последовательную обработку для стабильности
        sequential_processing = True
        
        # Параллельная обработка батча если возможно и размер достаточно большой
        if len(batch) >= 4 and N_JOBS > 1 and not sequential_processing:
            try:
                with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                    # Добавляем флаг for_emotion=True для извлечения признаков эмоций
                    features_list = list(executor.map(
                        lambda x: extract_features(x, for_emotion=True), 
                        batch
                    ))
                
                # Добавляем обработанный батч в датасет
                for features in features_list:
                    dataset.append({'features': features, 'label': emotion})
            except Exception as e:
                # Последовательное извлечение признаков при ошибке
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                
                error_logger.log_error(f"Параллельная обработка не удалась: {str(e)}, переключаемся на последовательную", 
                                     "processing", "dataset_creator")
                
                for fragment in batch:
                    try:
                        features = extract_features(fragment, for_emotion=True)
                        dataset.append({'features': features, 'label': emotion})
                    except Exception as inner_e:
                        error_logger.log_error(f"Ошибка при извлечении признаков: {str(inner_e)}", 
                                           "processing", "dataset_creator")
        else:
            # Последовательное извлечение признаков
            for fragment in batch:
                try:
                    features = extract_features(fragment, for_emotion=True)
                    dataset.append({'features': features, 'label': emotion})
                except Exception as e:
                    error_logger.log_error(f"Ошибка при извлечении признаков: {str(e)}", 
                                       "processing", "dataset_creator")
        
        # Принудительная сборка мусора после обработки каждого батча
        gc.collect()
    
    # Проверка наличия данных в датасете
    if not dataset:
        raise ValueError("Не удалось создать датасет: все операции извлечения признаков завершились с ошибками")
    
    return dataset

# Делаем функцию extract_features "безопасной" для многопроцессорной обработки
# Примечание: lru_cache не может быть использован напрямую с ProcessPoolExecutor
def extract_features(audio_data, for_emotion=False):
    """
    Извлекает признаки из аудиофрагмента для задачи идентификации голоса или распознавания эмоций.
    Оптимизированная версия с поддержкой параллельных вычислений для внутренних операций.
    Обеспечивает стандартную размерность выходных данных независимо от длины входного аудио.
    """
    try:
        # Проверка данных аудио
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Аудио данные отсутствуют или пусты")
            
        # Если аудио слишком короткое, применяем обработку вместо генерации ошибки
        min_audio_length = N_FFT * 2
        if len(audio_data) < min_audio_length:
            # Логируем информацию о коротком фрагменте
            error_logger.log_error(
                f"Обнаружен короткий аудиофрагмент: {len(audio_data)} < {min_audio_length}, применяем pad/repeat",
                "processing", "extract_features"
            )
            
            # Вариант 1: Применяем padding нулями до нужной длины
            padding_length = min_audio_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding_length), mode='wrap')
        
        # Жестко ограничиваем длину обработки для оптимизации вычислений и памяти
        max_samples = int(MAX_FEATURE_LENGTH * SAMPLE_RATE)
        if len(audio_data) > max_samples:
            error_logger.log_error(
                f"Длина аудио сокращена с {len(audio_data)} до {max_samples} отсчетов для экономии памяти",
                "processing", "extract_features"
            )
            audio_data = audio_data[:max_samples]
        
        # Преэмфазис для улучшения распознавания речи (быстрая реализация с numpy)
        emphasized_audio = np.append(
            audio_data[0], 
            audio_data[1:] - PREEMPHASIS_COEF * audio_data[:-1]
        )
        
        # Извлекаем основные спектральные признаки
        # Рассчитываем STFT один раз для использования во всех спектральных признаках
        stft = librosa.stft(emphasized_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        magnitude = np.abs(stft)
        power_spectrogram = magnitude ** 2
        
        # MFCC - основной набор признаков
        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(power_spectrogram),
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Добавляем дельта-коэффициенты для улучшения точности
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Спектральный центроид характеризует "яркость" звука
        centroid = librosa.feature.spectral_centroid(
            S=magnitude, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH
        )
        
        # Спектральный контраст характеризует разницу между пиками и впадинами
        # Уменьшаем количество полос для оптимизации
        n_bands = 4  # Уменьшено с 6 для оптимизации
        contrast = librosa.feature.spectral_contrast(
            S=magnitude, 
            n_bands=n_bands,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Для эмоций добавляем дополнительные признаки
        if for_emotion:
            # Частотный рулон (спектральный наклон) - показывает баланс высоких и низких частот
            rolloff = librosa.feature.spectral_rolloff(
                S=magnitude, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH
            )
            
            # Для коротких аудио упрощаем расчет фундаментальной частоты
            if len(audio_data) < SAMPLE_RATE * 1.5:  # Менее 1.5 сек
                # Упрощенный расчет F0 для коротких аудио
                f0 = librosa.yin(
                    emphasized_audio, 
                    fmin=librosa.note_to_hz('C2'), 
                    fmax=librosa.note_to_hz('C7'),
                    sr=SAMPLE_RATE,
                    hop_length=HOP_LENGTH * 2  # Используем большее значение hop_length для ускорения
                )
                # Преобразуем в формат признака
                f0 = f0.reshape(1, -1)
                # Приводим размерность к другим признакам (если нужно)
                if f0.shape[1] != mfcc.shape[1]:
                    f0 = _fast_resize_feature(f0, mfcc.shape[1])
            else:
                # Полный расчет F0 для длинных аудио
                f0 = librosa.yin(
                    emphasized_audio, 
                    fmin=librosa.note_to_hz('C2'), 
                    fmax=librosa.note_to_hz('C7'),
                    sr=SAMPLE_RATE,
                    hop_length=HOP_LENGTH
                )
                f0 = f0.reshape(1, -1)
            
            # RMS энергия для определения громкости
            rms = librosa.feature.rms(
                S=magnitude,
                hop_length=HOP_LENGTH
            )
        
        # Функция быстрой нормализации с помощью numpy
        def fast_normalize(feature_matrix):
            mean = np.mean(feature_matrix, axis=1, keepdims=True)
            std = np.std(feature_matrix, axis=1, keepdims=True) + 1e-5  # Избегаем деления на ноль
            return (feature_matrix - mean) / std
        
        # Объединяем признаки
        if for_emotion:
            # Нормализуем все признаки
            mfcc_normalized = fast_normalize(mfcc)
            mfcc_delta_normalized = fast_normalize(mfcc_delta)
            mfcc_delta2_normalized = fast_normalize(mfcc_delta2)
            centroid_normalized = fast_normalize(centroid)
            contrast_normalized = fast_normalize(contrast)
            rolloff_normalized = fast_normalize(rolloff)
            f0_normalized = fast_normalize(f0)
            rms_normalized = fast_normalize(rms)
            
            # Объединяем все признаки в одну матрицу с помощью np.vstack
            features = np.vstack([
                mfcc_normalized, 
                mfcc_delta_normalized, 
                mfcc_delta2_normalized,
                centroid_normalized,
                contrast_normalized,
                rolloff_normalized,
                f0_normalized,
                rms_normalized
            ])
        else:
            # Для идентификации голоса используем меньший набор признаков
            # Нормализуем признаки
            mfcc_normalized = fast_normalize(mfcc)
            mfcc_delta_normalized = fast_normalize(mfcc_delta)
            mfcc_delta2_normalized = fast_normalize(mfcc_delta2)
            centroid_normalized = fast_normalize(centroid)
            contrast_normalized = fast_normalize(contrast)
            
            # Объединяем признаки
            features = np.vstack([
                mfcc_normalized, 
                mfcc_delta_normalized, 
                mfcc_delta2_normalized,
                centroid_normalized,
                contrast_normalized
            ])
        
        # Транспонируем для соответствия формату (n_samples, n_features)
        features = features.T
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Стандартизируем размерность признаков, чтобы гарантировать фиксированный размер
        # независимо от длины входного аудио
        num_frames = features.shape[0]
        num_features = features.shape[1]
        
        if num_frames > MAX_FRAMES:
            # Если фреймов больше максимума, выбираем равномерно распределенные фреймы
            indices = np.linspace(0, num_frames - 1, MAX_FRAMES, dtype=int)
            features = features[indices]
            error_logger.log_error(
                f"Количество фреймов признаков сокращено с {num_frames} до {MAX_FRAMES}",
                "processing", "extract_features"
            )
        elif num_frames < MAX_FRAMES:
            # Если фреймов меньше максимума, дополняем нулями или повторяем
            padding = np.zeros((MAX_FRAMES - num_frames, num_features))
            features = np.vstack([features, padding])
            error_logger.log_error(
                f"Количество фреймов признаков увеличено с {num_frames} до {MAX_FRAMES} путем дополнения",
                "processing", "extract_features"
            )
        
        # Теперь features имеет фиксированную форму (MAX_FRAMES, n_features)
        return features
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_message = f"Ошибка при извлечении признаков: {str(e)}"
        error_logger.log_error(error_message, "features", "extract_features")
        raise ValueError(error_message)

def _fast_resize_feature(feature, target_length):
    """
    Быстрое изменение размера матрицы признаков до целевой длины
    Использует эффективные операции numpy
    """
    current_length = feature.shape[1]
    
    if current_length == target_length:
        return feature
    
    if current_length > target_length:
        # Если текущая длина больше, отрезаем лишнее
        return feature[:, :target_length]
    else:
        # Если текущая длина меньше, дополняем нулями
        padding = np.zeros((feature.shape[0], target_length - current_length))
        return np.hstack((feature, padding))
