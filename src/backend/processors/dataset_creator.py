import numpy as np
import librosa
from .augmentation_processor import augment_audio
from backend.api.error_logger import error_logger
from backend.config import (
    DATASET_CREATOR, SAMPLE_RATE,
    HOP_LENGTH, DATASET_CREATOR
)
import random
import sys
import os

# Используем константы из конфигурационного файла
N_MFCC = DATASET_CREATOR['N_MFCC']
MAX_FRAMES = DATASET_CREATOR['MAX_FRAMES']
EMOTIONS = DATASET_CREATOR['EMOTIONS']
PREEMPHASIS_COEF = DATASET_CREATOR['PREEMPHASIS_COEF']
N_FFT_DATASET = DATASET_CREATOR['N_FFT_DATASET']



def create_voice_id_dataset(audio_fragments, name):
    """
    Создание датасета для модели идентификации по голосу
    с параллельной обработкой фрагментов
    """
    if not audio_fragments:
        error_info = error_logger.log_exception(
            ValueError("Нет аудиофрагментов для создания датасета"),
            "dataset_creator",
            "validation",
            "Проверка входных данных"
        )
        raise ValueError("Нет аудиофрагментов для создания датасета")
    
    # Аугментация аудиофрагментов с ограничением результатов
    try:
        augmented_fragments = augment_audio(audio_fragments)
    except Exception as e:
        error_info = error_logger.log_exception(
            e,
            "dataset_creator",
            "augmentation",
            "Ошибка при аугментации. Продолжаем без аугментации."
        )
        # Если аугментация не удалась, используем только исходные фрагменты
        augmented_fragments = audio_fragments.copy()

    # Создание датасета для обучения нейронной сети
    # Используем пакетную обработку вместо обработки всего сразу
    dataset = []
    
    for i in range(0, len(augmented_fragments), batch_size):
        batch = augmented_fragments[i:i+batch_size]
        
        # ИЗМЕНЕНО: Предпочитаем последовательную обработку для стабильности
        sequential_processing = True
        
        # Параллельная обработка батча если возможно и размер достаточно большой
        if len(batch) >= 4 and not sequential_processing:
            try:
                features_list = [extract_features(fragment) for fragment in batch]
                
                # Добавляем обработанный батч в датасет
                for features in features_list:
                    dataset.append({'features': features, 'label': name})
            except Exception as e:
                # Последовательное извлечение признаков при ошибке
                for fragment in batch:
                    try:
                        features = extract_features(fragment)
                        dataset.append({'features': features, 'label': name})
                    except Exception as e:
                        pass
        else:
            # Последовательное извлечение признаков
            for fragment in batch:
                try:
                    features = extract_features(fragment)
                    dataset.append({'features': features, 'label': name})
                except Exception as e:
                    pass
    
    # Проверка наличия данных в датасете
    if not dataset:
        error_info = error_logger.log_exception(
            ValueError("Не удалось создать датасет: все операции извлечения признаков завершились с ошибками"),
            "dataset_creator",
            "validation",
            "Проверка результатов обработки данных"
        )
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
        error_info = error_logger.log_exception(
            ValueError("Нет аудиофрагментов для создания датасета"),
            "dataset_creator",
            "validation",
            "Проверка входных данных"
        )
        raise ValueError("Нет аудиофрагментов для создания датасета")
    
    if emotion not in EMOTIONS:
        error_info = error_logger.log_exception(
            ValueError(f"Недопустимая эмоция. Допустимые: {', '.join(EMOTIONS)}"),
            "dataset_creator",
            "validation",
            "Проверка эмоции"
        )
        raise ValueError(f"Недопустимая эмоция. Допустимые: {', '.join(EMOTIONS)}")
    
    
    # Аугментация аудиофрагментов с ограничением результатов
    try:
        augmented_fragments = augment_audio(audio_fragments)
    except Exception as e:
        error_info = error_logger.log_exception(
            e,
            "dataset_creator",
            "augmentation",
            "Ошибка при аугментации. Продолжаем без аугментации."
        )
        # Если аугментация не удалась, используем только исходные фрагменты
        augmented_fragments = audio_fragments.copy()
    
    # Создание датасета для обучения нейронной сети
    # Используем пакетную обработку вместо обработки всего сразу
    dataset = []
    
    for i in range(0, len(augmented_fragments), batch_size):
        batch = augmented_fragments[i:i+batch_size]
        
        # Параллельная обработка батча если возможно и размер достаточно большой
        if len(batch) >= 4:
            try:
                features_list = [extract_features(fragment, for_emotion=True) for fragment in batch]
                
                # Эмоции кодируем как индексы
                emotion_map = {'гнев': 0, 'радость': 1, 'грусть': 2}
                emotion_idx = emotion_map[emotion]
                
                # Добавляем обработанный батч в датасет
                for features in features_list:
                    if features is not None:
                        dataset.append({'features': features, 'label': emotion_idx})
            except Exception as e:
                # Последовательное извлечение признаков при ошибке
                emotion_map = {'гнев': 0, 'радость': 1, 'грусть': 2}
                emotion_idx = emotion_map[emotion]
                
                for fragment in batch:
                    try:
                        features = extract_features(fragment, for_emotion=True)
                        if features is not None:
                            dataset.append({'features': features, 'label': emotion_idx})
                    except Exception as e:
                        pass
        else:
            # Последовательное извлечение признаков
            emotion_map = {'гнев': 0, 'радость': 1, 'грусть': 2}
            emotion_idx = emotion_map[emotion]
            
            for fragment in batch:
                try:
                    features = extract_features(fragment, for_emotion=True)
                    if features is not None:
                        dataset.append({'features': features, 'label': emotion_idx})
                except Exception as e:
                    pass
        
    
    # Проверка наличия данных в датасете
    if not dataset:
        error_info = error_logger.log_exception(
            ValueError("Не удалось создать датасет для эмоций: все операции извлечения признаков завершились с ошибками"),
            "dataset_creator",
            "validation",
            "Проверка результатов обработки данных"
        )
        raise ValueError("Не удалось создать датасет для эмоций: все операции извлечения признаков завершились с ошибками")
    
    return dataset

def extract_features(audio_data, for_emotion=False):
    """
    Извлечение признаков из аудиоданных для задач распознавания речи и эмоций.
    
    Args:
        audio_data: numpy массив с аудиоданными
        for_emotion: флаг, указывающий, извлекаются ли признаки для распознавания эмоций
            
    Returns:
        numpy массив с извлеченными признаками
    """
    try:
        # Проверка данных аудио
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Аудио данные отсутствуют или пусты")
            
        # Если аудио слишком короткое, применяем обработку вместо генерации ошибки
        min_audio_length = N_FFT_DATASET * 2
        if len(audio_data) < min_audio_length:
            # Логируем информацию о коротком фрагменте
            error_logger.log_error(
                f"Обнаружен короткий аудиофрагмент: {len(audio_data)} < {min_audio_length}, применяем pad/repeat",
                "dataset_creator", "extract_features"
            )
            
            # Вариант 1: Применяем padding нулями до нужной длины
            padding_length = min_audio_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding_length), mode='wrap')
        
        # Предобработка: предэмфазис для усиления высоких частот
        emphasized_audio = np.append(
            audio_data[0], 
            audio_data[1:] - PREEMPHASIS_COEF * audio_data[:-1]
        )
        
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
        # Уменьшаем количество полос для оптимизации
        n_bands = 4  # Уменьшено с 6 для оптимизации
        contrast = librosa.feature.spectral_contrast(
            S=magnitude, 
            n_bands=n_bands,
            n_fft=N_FFT_DATASET,
            hop_length=HOP_LENGTH
        )
        
        # Функция быстрой нормализации с помощью numpy
        def fast_normalize(feature_matrix):
            mean = np.mean(feature_matrix, axis=1, keepdims=True)
            std = np.std(feature_matrix, axis=1, keepdims=True) + 1e-5  # Избегаем деления на ноль
            return (feature_matrix - mean) / std
        
        # Для эмоций добавляем дополнительные признаки
        if for_emotion:
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
            mfcc_normalized = fast_normalize(mfcc)
            mfcc_delta_normalized = fast_normalize(mfcc_delta)
            mfcc_delta2_normalized = fast_normalize(mfcc_delta2)
            centroid_normalized = fast_normalize(centroid)
            contrast_normalized = fast_normalize(contrast)
            rolloff_normalized = fast_normalize(rolloff)
            f0_normalized = fast_normalize(f0)
            rms_normalized = fast_normalize(rms)
            chroma_normalized = fast_normalize(chroma)
            
            # Объединяем все признаки в одну матрицу с помощью np.vstack
            features = np.vstack([
                mfcc_normalized, 
                mfcc_delta_normalized, 
                mfcc_delta2_normalized,
                centroid_normalized,
                contrast_normalized,
                rolloff_normalized,
                f0_normalized,
                rms_normalized,
                chroma_normalized
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
        
        # УЛУЧШЕНИЕ: Более надежная стандартизация размерности признаков
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
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        return None

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
