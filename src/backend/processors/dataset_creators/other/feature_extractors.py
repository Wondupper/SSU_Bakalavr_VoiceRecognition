import numpy as np
import librosa
from backend.config import (
    DATASET_CREATOR, SAMPLE_RATE,
    HOP_LENGTH
)
from backend.api.error_logger import error_logger

# Используем константы из конфигурационного файла
N_MFCC = DATASET_CREATOR['N_MFCC']
MAX_FRAMES = DATASET_CREATOR['MAX_FRAMES']
PREEMPHASIS_COEF = DATASET_CREATOR['PREEMPHASIS_COEF']
N_FFT_DATASET = DATASET_CREATOR['N_FFT_DATASET']

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

def _fast_normalize(feature_matrix):
    """
    Быстрая нормализация с помощью numpy
    """
    mean = np.mean(feature_matrix, axis=1, keepdims=True)
    std = np.std(feature_matrix, axis=1, keepdims=True) + 1e-5  # Избегаем деления на ноль
    return (feature_matrix - mean) / std

def _preprocess_audio(audio_data):
    """
    Общая предобработка аудиоданных для извлечения признаков
    """
    # Проверка данных аудио
    if audio_data is None or len(audio_data) == 0:
        raise ValueError("Аудио данные отсутствуют или пусты")
        
    # Если аудио слишком короткое, применяем обработку вместо генерации ошибки
    min_audio_length = N_FFT_DATASET * 2
    if len(audio_data) < min_audio_length:
        # Применяем padding нулями до нужной длины
        padding_length = min_audio_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding_length), mode='wrap')
    
    # Предобработка: предэмфазис для усиления высоких частот
    emphasized_audio = np.append(
        audio_data[0], 
        audio_data[1:] - PREEMPHASIS_COEF * audio_data[:-1]
    )
    
    return emphasized_audio

def _standardize_features(features):
    """
    Стандартизация размерности признаков до MAX_FRAMES
    """
    # Транспонируем для соответствия формату (n_samples, n_features)
    features = features.T
    
    # Превращаем в форму (MAX_FRAMES, n_features)
    num_frames = features.shape[0]
    
    # ГАРАНТИРУЕМ фиксированный размер кадров (MAX_FRAMES)
    if num_frames > MAX_FRAMES:
        # Если фреймов больше максимума, выбираем равномерно распределенные фреймы
        indices = np.linspace(0, num_frames - 1, MAX_FRAMES, dtype=int)
        features = features[indices]
    elif num_frames < MAX_FRAMES:
        # Если фреймов меньше максимума, дополняем нулями
        padding = np.zeros((MAX_FRAMES - num_frames, features.shape[1]))
        features = np.vstack([features, padding])
    
    # Проверка на некорректные значения (NaN, Inf)
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features)
    
    return features

def extract_voice_id_features(audio_data):
    """
    Извлечение признаков из аудиоданных для задач идентификации по голосу.
    
    Args:
        audio_data: numpy массив с аудиоданными
            
    Returns:
        numpy массив с извлеченными признаками размерности (MAX_FRAMES, n_features), где:
        - MAX_FRAMES: фиксированное количество фреймов (задано в конфигурации)
        - n_features: количество признаков для каждого фрейма (N_MFCC*3 + 1 + n_bands)
    """
    try:
        emphasized_audio = _preprocess_audio(audio_data)
        
        # Извлекаем основные спектральные признаки
        # Рассчитываем STFT один раз для использования во всех спектральных признаках
        stft = librosa.stft(emphasized_audio, n_fft=N_FFT_DATASET, hop_length=HOP_LENGTH)
        magnitude = np.abs(stft)
        power_spectrogram = magnitude ** 2
        
        # MFCC - основной набор признаков
        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(power_spectrogram),
            n_mfcc=N_MFCC,
            n_fft=N_FFT_DATASET,
            hop_length=HOP_LENGTH
        )
        
        # Добавляем дельта-коэффициенты для улучшения точности
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Спектральный центроид характеризует "яркость" звука
        centroid = librosa.feature.spectral_centroid(
            S=magnitude, 
            n_fft=N_FFT_DATASET, 
            hop_length=HOP_LENGTH
        )
        
        # Спектральный контраст характеризует разницу между пиками и впадинами
        n_bands = 4  # Уменьшено с 6 для оптимизации
        contrast = librosa.feature.spectral_contrast(
            S=magnitude, 
            n_bands=n_bands,
            n_fft=N_FFT_DATASET,
            hop_length=HOP_LENGTH
        )
        
        # Для идентификации по голосу используем меньший набор признаков
        # Нормализуем признаки
        mfcc_normalized = _fast_normalize(mfcc)
        mfcc_delta_normalized = _fast_normalize(mfcc_delta)
        mfcc_delta2_normalized = _fast_normalize(mfcc_delta2)
        centroid_normalized = _fast_normalize(centroid)
        contrast_normalized = _fast_normalize(contrast)
        
        # Объединяем признаки
        features = np.vstack([
            mfcc_normalized,                # 40 (N_MFCC) признаков
            mfcc_delta_normalized,          # 40 (N_MFCC) признаков
            mfcc_delta2_normalized,         # 40 (N_MFCC) признаков
            centroid_normalized,            # 1 признак
            contrast_normalized             # 4 (n_bands) признака
        ])
        # Итого для идентификации: 40*3 + 1 + 4 = 125 признаков
        
        return _standardize_features(features)
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "feature_extractors.py",
            "extract_voice_id_features",
            "Ошибка при извлечении признаков для идентификации голоса"
        )
        return None

def extract_emotion_features(audio_data):
    """
    Извлечение признаков из аудиоданных для задач распознавания эмоций.
    
    Args:
        audio_data: numpy массив с аудиоданными
            
    Returns:
        numpy массив с извлеченными признаками размерности (MAX_FRAMES, n_features), где:
        - MAX_FRAMES: фиксированное количество фреймов (задано в конфигурации)
        - n_features: количество признаков для каждого фрейма (N_MFCC*3 + 1 + n_bands + 1 + 1 + 1 + 12)
    """
    try:
        emphasized_audio = _preprocess_audio(audio_data)
        
        # Извлекаем основные спектральные признаки
        # Рассчитываем STFT один раз для использования во всех спектральных признаках
        stft = librosa.stft(emphasized_audio, n_fft=N_FFT_DATASET, hop_length=HOP_LENGTH)
        magnitude = np.abs(stft)
        power_spectrogram = magnitude ** 2
        
        # MFCC - основной набор признаков
        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(power_spectrogram),
            n_mfcc=N_MFCC,
            n_fft=N_FFT_DATASET,
            hop_length=HOP_LENGTH
        )
        
        # Добавляем дельта-коэффициенты для улучшения точности
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Спектральный центроид характеризует "яркость" звука
        centroid = librosa.feature.spectral_centroid(
            S=magnitude, 
            n_fft=N_FFT_DATASET, 
            hop_length=HOP_LENGTH
        )
        
        # Спектральный контраст характеризует разницу между пиками и впадинами
        n_bands = 4  # Уменьшено с 6 для оптимизации
        contrast = librosa.feature.spectral_contrast(
            S=magnitude, 
            n_bands=n_bands,
            n_fft=N_FFT_DATASET,
            hop_length=HOP_LENGTH
        )
        
        # Частотный рулон (спектральный наклон) - показывает баланс высоких и низких частот
        rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, 
            n_fft=N_FFT_DATASET, 
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
        
        # Получаем хроматограмму (для моделей эмоций)
        chroma = librosa.feature.chroma_stft(
            y=emphasized_audio, 
            sr=SAMPLE_RATE,
            n_fft=N_FFT_DATASET,
            hop_length=HOP_LENGTH
        )
        
        # Нормализуем все признаки
        mfcc_normalized = _fast_normalize(mfcc)
        mfcc_delta_normalized = _fast_normalize(mfcc_delta)
        mfcc_delta2_normalized = _fast_normalize(mfcc_delta2)
        centroid_normalized = _fast_normalize(centroid)
        contrast_normalized = _fast_normalize(contrast)
        rolloff_normalized = _fast_normalize(rolloff)
        f0_normalized = _fast_normalize(f0)
        rms_normalized = _fast_normalize(rms)
        chroma_normalized = _fast_normalize(chroma)
        
        # Объединяем все признаки в одну матрицу с помощью np.vstack
        features = np.vstack([
            mfcc_normalized,                # 40 (N_MFCC) признаков
            mfcc_delta_normalized,          # 40 (N_MFCC) признаков
            mfcc_delta2_normalized,         # 40 (N_MFCC) признаков
            centroid_normalized,            # 1 признак
            contrast_normalized,            # 4 (n_bands) признака
            rolloff_normalized,             # 1 признак
            f0_normalized,                  # 1 признак
            rms_normalized,                 # 1 признак
            chroma_normalized               # 12 признаков
        ])
        # Итого для эмоций: 40*3 + 1 + 4 + 1 + 1 + 1 + 12 = 140 признаков
        
        return _standardize_features(features)
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "feature_extractors.py",
            "extract_emotion_features",
            "Ошибка при извлечении признаков для распознавания эмоций"
        )
        return None 