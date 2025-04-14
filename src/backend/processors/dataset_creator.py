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

# Константы для вычисления признаков
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
PREEMPHASIS_COEF = 0.97
MAX_FEATURE_LENGTH = 5  # Макс. длительность в секундах для экстракции признаков
SAMPLE_RATE = 16000     # Частота дискретизации

# Определяем оптимальное количество процессов
# Оставляем 1 ядро для основных операций системы
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

def create_voice_id_dataset(audio_fragments, name):
    """
    Создание датасета для модели идентификации по голосу
    с параллельной обработкой фрагментов
    """
    if not audio_fragments:
        raise ValueError("Нет аудиофрагментов для создания датасета")
    
    # Аугментация аудиофрагментов
    augmented_fragments = augment_audio(audio_fragments)
    
    # Создание датасета для обучения нейронной сети
    # Оптимизация: предварительно выделяем память для датасета
    dataset_size = len(augmented_fragments)
    
    # Используем параллельную обработку для извлечения признаков
    # только если у нас достаточно данных для распараллеливания
    if dataset_size >= 4 and N_JOBS > 1:
        # Параллельное извлечение признаков
        try:
            with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                # Параллельно запускаем извлечение признаков
                features_list = list(executor.map(extract_features, augmented_fragments))
            
            # Создаем датасет из полученных признаков
            dataset = [{'features': features, 'label': name} for features in features_list]
        except Exception as e:
            # Если параллельное извлечение не удалось, используем последовательное
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            
            error_logger.log_error(f"Параллельная обработка не удалась: {str(e)}, переключаемся на последовательную", 
                                  "processing", "dataset_creator")
            dataset = [{'features': extract_features(fragment), 'label': name} 
                      for fragment in augmented_fragments]
    else:
        # Последовательное извлечение признаков для небольших наборов данных
        dataset = [{'features': extract_features(fragment), 'label': name} 
                  for fragment in augmented_fragments]
    
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
    
    # Аугментация аудиофрагментов
    augmented_fragments = augment_audio(audio_fragments)
    
    # Создание датасета для обучения нейронной сети
    dataset_size = len(augmented_fragments)
    
    # Используем параллельную обработку для извлечения признаков
    # только если у нас достаточно данных для распараллеливания
    if dataset_size >= 4 and N_JOBS > 1:
        # Параллельное извлечение признаков
        try:
            with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
                # Параллельно запускаем извлечение признаков
                features_list = list(executor.map(extract_features, augmented_fragments))
            
            # Создаем датасет из полученных признаков
            dataset = [{'features': features, 'label': emotion} for features in features_list]
        except Exception as e:
            # Если параллельное извлечение не удалось, используем последовательное
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            
            error_logger.log_error(f"Параллельная обработка не удалась: {str(e)}, переключаемся на последовательную", 
                                  "processing", "dataset_creator")
            dataset = [{'features': extract_features(fragment), 'label': emotion} 
                      for fragment in augmented_fragments]
    else:
        # Последовательное извлечение признаков для небольших наборов данных
        dataset = [{'features': extract_features(fragment), 'label': emotion} 
                  for fragment in augmented_fragments]
    
    return dataset

# Делаем функцию extract_features "безопасной" для многопроцессорной обработки
# Примечание: lru_cache не может быть использован напрямую с ProcessPoolExecutor
def extract_features(audio_data, for_emotion=False):
    """
    Извлекает признаки из аудиофрагмента для задачи идентификации голоса или распознавания эмоций.
    Оптимизированная версия с поддержкой параллельных вычислений для внутренних операций.
    """
    try:
        # Проверка данных аудио
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Аудио данные отсутствуют или пусты")
            
        # Если аудио слишком короткое, возвращаем ошибку
        min_audio_length = N_FFT * 2
        if len(audio_data) < min_audio_length:
            raise ValueError(f"Длина аудио слишком мала для извлечения признаков: {len(audio_data)} < {min_audio_length}")
            
        # Ограничиваем длину обработки для оптимизации вычислений
        max_samples = int(MAX_FEATURE_LENGTH * SAMPLE_RATE)
        if len(audio_data) > max_samples:
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
        
        # Объединяем признаки
        if for_emotion:
            # Используем векторизованные операции для нормализации
            # вместо циклов по каждому признаку
            
            # Функция быстрой нормализации с помощью numpy
            def fast_normalize(feature_matrix):
                mean = np.mean(feature_matrix, axis=1, keepdims=True)
                std = np.std(feature_matrix, axis=1, keepdims=True) + 1e-5  # Избегаем деления на ноль
                return (feature_matrix - mean) / std
            
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
            # Используем векторизованные операции для нормализации
            
            def fast_normalize(feature_matrix):
                mean = np.mean(feature_matrix, axis=1, keepdims=True)
                std = np.std(feature_matrix, axis=1, keepdims=True) + 1e-5
                return (feature_matrix - mean) / std
            
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
        
        return features.T  # Транспонируем для соответствия формату (n_samples, n_features)
        
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
