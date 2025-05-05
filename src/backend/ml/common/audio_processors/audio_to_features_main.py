import torch
from typing import List, Optional, Tuple
from werkzeug.datastructures import FileStorage
from src.backend.ml.common.augmentations.augmentator_main import apply_augmentation
from src.backend.ml.common.audio_processors.processors.audio_loader import load_audio_from_file
from src.backend.ml.common.audio_processors.processors.audio_processor import preprocess_audio
from src.backend.ml.common.audio_processors.processors.noise_reductor import apply_noise_reduction
from src.backend.ml.common.audio_processors.processors.silence_remover import remove_silence
from src.backend.ml.common.audio_processors.processors.splitter import split_into_fragments
from src.backend.ml.common.audio_processors.processors.mfcc_extractor import extract_mfcc_features
from src.backend.ml.common.audio_processors.processors.delta_features_computes import compute_delta_features
from src.backend.ml.common.audio_processors.processors.spectral_features_extractor import extract_spectral_features
from src.backend.ml.common.audio_processors.processors.features_combainer import combine_features


def get_features_tensors_from_audio(audio_file: FileStorage, target_length: int) -> List[torch.Tensor]:
    """
    Извлекает признаки из аудиофайла с помощью torchaudio
    
    Args:
        audio_file: Файл аудио (объект FileStorage Flask)
        target_length: Целевая длина тензора признаков
        
    Returns:
        Список тензоров признаков для каждого фрагмента
    """
    # 1. Загрузка аудио из файла
    waveform, sample_rate = load_audio_from_file(audio_file)
    
    # 2. Предварительная обработка
    waveform = preprocess_audio(waveform, sample_rate)
    
    # 3. Применение шумоподавления
    enhanced_waveform = apply_noise_reduction(waveform)
    
    # 4. Удаление тишины
    enhanced_waveform = remove_silence(enhanced_waveform)
    
    # 5. Применение аугментации
    augmented_waveforms = apply_augmentation(enhanced_waveform)
    
    # 6. Обработка каждой аугментированной формы
    features_list = []
    for aug_waveform in augmented_waveforms:
        # 7. Разбиение на фрагменты
        fragments = split_into_fragments(aug_waveform)
        
        # 8. Извлечение признаков из каждого фрагмента
        for fragment in fragments:

            # 1. MFCC признаки
            mfcc = extract_mfcc_features(fragment)

            # 2. Дельта и дельта-дельта коэффициенты
            delta, delta2 = compute_delta_features(mfcc)

            # 3. Спектральные признаки
            spec_features = extract_spectral_features(fragment, mfcc.shape[2])

            # 4. Объединяем все признаки
            features = combine_features(mfcc, delta, delta2, spec_features, target_length)
            
            features_list.append(features)
    
    return features_list
        