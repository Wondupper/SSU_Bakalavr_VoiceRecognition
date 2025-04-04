import numpy as np
import librosa
from .augmentation_processor import augment_audio

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

def create_emotion_dataset(audio_fragments):
    """
    Создание датасета для модели распознавания эмоций
    """
    if not audio_fragments:
        raise ValueError("Нет аудиофрагментов для создания датасета")
    
    # Аугментация аудиофрагментов
    augmented_fragments = augment_audio(audio_fragments)
    
    # Создание датасета для обучения нейронной сети
    dataset = []
    
    for fragment in augmented_fragments:
        # Извлечение признаков из аудиофрагмента
        features = extract_features(fragment)
        
        # Для автоматического определения эмоции используем метки без указания конкретной эмоции
        # Модель сама будет учиться определять эмоциональные паттерны
        dataset.append({
            'features': features
        })
    
    return dataset

def extract_features(audio_data, sr=16000):
    """
    Извлечение признаков из аудиоданных для обучения нейронной сети
    """
    # Извлечение MFCC (мел-кепстральных коэффициентов)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    
    # Нормализация признаков
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    # Добавление дополнительных признаков
    
    # Спектральный центроид
    centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    centroid = (centroid - np.mean(centroid)) / np.std(centroid)
    
    # Спектральная контрастность
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    contrast = (contrast - np.mean(contrast)) / np.std(contrast)
    
    # Объединение признаков
    features = np.vstack([mfccs, centroid, contrast])
    
    return features
