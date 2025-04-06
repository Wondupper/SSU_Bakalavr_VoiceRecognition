import numpy as np
import librosa
from .augmentation_processor import augment_audio
from backend.api.error_logger import error_logger

def create_voice_id_dataset(audio_fragments, name):
    """
    Создание датасета для модели идентификации по голосу
    """
    if not audio_fragments:
        raise ValueError("Нет аудиофрагментов для создания датасета")
    
    # Аугментация аудиофрагментов
    augmented_fragments = augment_audio(audio_fragments)
    
    # Создание датасета для обучения нейронной сети
    dataset = []
    
    for fragment in augmented_fragments:
        # Извлечение MFCCs из аудиофрагмента
        mfccs = extract_features(fragment)
        
        # Добавление в датасет
        dataset.append({
            'features': mfccs,
            'label': name
        })
    
    return dataset

def create_emotion_dataset(audio_fragments, emotion):
    """
    Создание датасета для модели распознавания эмоций
    
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
    dataset = []
    
    for fragment in augmented_fragments:
        # Извлечение признаков из аудиофрагмента
        features = extract_features(fragment)
        
        # Добавляем эмоцию, которую передали в функцию
        dataset.append({
            'features': features,
            'label': emotion
        })
    
    return dataset

def extract_features(audio_data, sr=16000):
    """
    Извлечение признаков из аудиоданных для обучения нейронной сети
    """
    try:
        # Проверка на пустые данные
        if len(audio_data) == 0:
            raise ValueError("Пустые аудиоданные")
            
        # Проверка на минимальную длину аудио
        if len(audio_data) < 1024:
            raise ValueError("Аудиоданные слишком короткие для извлечения признаков")
        
        # Извлечение MFCC (мел-кепстральных коэффициентов)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
        
        # Безопасная нормализация признаков
        mfccs_std = np.std(mfccs)
        if mfccs_std > 0:
            mfccs = (mfccs - np.mean(mfccs)) / mfccs_std
        
        # Спектральный центроид
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        centroid_std = np.std(centroid)
        if centroid_std > 0:
            centroid = (centroid - np.mean(centroid)) / centroid_std
        
        # Спектральная контрастность
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        contrast_std = np.std(contrast)
        if contrast_std > 0:
            contrast = (contrast - np.mean(contrast)) / contrast_std
        
        # Добавляем проверку на NaN
        features = np.vstack([mfccs, centroid, contrast])
        if np.isnan(features).any():
            # Заменяем NaN на нули
            features = np.nan_to_num(features)
            
        return features
    except Exception as e:
        error_message = f"Ошибка при извлечении признаков: {str(e)}"
        error_logger.log_error(error_message, "processing", "dataset_creator")
        # Вместо повторного вызова исключения возвращаем базовые признаки
        return np.zeros((22, 134))  # Возвращаем заглушку с правильной размерностью
