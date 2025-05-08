import torch
import torchaudio
import io
from typing import List, Optional, Tuple
from werkzeug.datastructures import FileStorage
from backend.loggers.error_logger import error_logger
from backend.loggers.info_logger import info_logger
from backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH, IS_AUGMENTATION_ENABLED
from backend.ml.common.augmentator import apply_augmentation


def get_features_tensors_from_audio_for_training(audio_file: FileStorage, target_length: int) -> List[torch.Tensor]:
    """
    Извлекает признаки из аудиофайла с помощью torchaudio
    
    Args:
        audio_file: Файл аудио (объект FileStorage Flask)
        target_length: Целевая длина тензора признаков
        
    Returns:
        Список тензоров признаков для каждого фрагмента
    """
    
    enhanced_waveform: torch.Tensor = preprocess(audio_file=audio_file)
    
    # 5. Применение аугментации
    augmented_waveforms: List[torch.Tensor] = apply_augmentation(enhanced_waveform) if IS_AUGMENTATION_ENABLED else [enhanced_waveform]
    
    # 6. Обработка каждой аугментированной формы
    features_list: List[torch.Tensor] = []
    for aug_waveform in augmented_waveforms:
        features_list.extend(extract(aug_waveform, target_length)) 
    
    return features_list

def get_features_tensors_from_audio_for_prediction(audio_file: FileStorage, target_length: int) -> List[torch.Tensor]:
    """
    Извлекает признаки из аудиофайла с помощью torchaudio для предсказания
    
    Args:
        audio_file: Файл аудио (объект FileStorage Flask)
        target_length: Целевая длина тензора признаков
        
    Returns:
        Список тензоров признаков для каждого фрагмента
    """
    
    enhanced_waveform: torch.Tensor = preprocess(audio_file=audio_file)
    
    # 5. Обработка аудиоформы (без аугментации)
    features_list: List[torch.Tensor] = extract(enhanced_waveform, target_length)
    
    return features_list


def preprocess(audio_file: FileStorage) -> torch.Tensor:
    # 1. Загрузка аудио из файла
    waveform, sample_rate = load_audio_from_file(audio_file)
    
    # 2. Предварительная обработка
    waveform: torch.Tensor = preprocess_audio(waveform, sample_rate)
    
    # 3. Применение шумоподавления
    enhanced_waveform: torch.Tensor = apply_noise_reduction(waveform)
    
    # 4. Удаление тишины
    enhanced_waveform: torch.Tensor = remove_silence(enhanced_waveform)

    return enhanced_waveform


def extract(waveform: torch.Tensor, target_length: int) -> List[torch.Tensor]:
    try:
        features_list: List[torch.Tensor] = []
        
        # 6. Разбиение на фрагменты
        fragments: List[torch.Tensor] = split_into_fragments(waveform)
        
        # 7. Извлечение признаков из каждого фрагмента
        for fragment in fragments:
            # 1. MFCC признаки
            mfcc: torch.Tensor = extract_mfcc_features(fragment)

            # 2. Дельта и дельта-дельта коэффициенты
            delta: torch.Tensor
            delta2: torch.Tensor
            delta, delta2 = compute_delta_features(mfcc)

            # 3. Спектральные признаки
            spec_features: torch.Tensor = extract_spectral_features(fragment, mfcc.shape[2])

            # 4. Объединяем все признаки
            features: torch.Tensor = combine_features(mfcc, delta, delta2, spec_features, target_length)
            
            features_list.append(features)
        
        return features_list

    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "extract",
            "Ошибка при исполнении последовательности в процессе извлечения признаков"
        )
        return []


def load_audio_from_file(audio_file: FileStorage) -> Tuple[torch.Tensor, int]:
    """
    Загружает аудио из файла и возвращает wavform и sample_rate
    
    Args:
        audio_file: Файл аудио (объект FileStorage Flask)
    
    Returns:
        Tuple с waveform и sample_rate
    """
    try:
        # Сохраняем содержимое файла в буфер
        audio_buffer: io.BytesIO = io.BytesIO(audio_file.read())
        # Сбрасываем указатель в начало буфера
        audio_buffer.seek(0)
        # Загружаем аудио из буфера
        waveform: torch.Tensor
        sample_rate: int
        waveform, sample_rate = torchaudio.load(audio_buffer)
        # Сбрасываем указатель файла на начало для возможного дальнейшего использования
        audio_file.seek(0)
        
        return waveform, sample_rate
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "load_audio_from_file",
            "Ошибка при загрузке аудио из аудиофайла"
        )
        return ()
    

def preprocess_audio(waveform: torch.Tensor, original_sample_rate: int) -> torch.Tensor:
    """
    Выполняет предварительную обработку аудио: ресемплинг, преобразование в моно, нормализация
    
    Args:
        waveform: Тензор аудиоформы
        original_sample_rate: Исходная частота дискретизации
    
    Returns:
        Обработанный тензор аудиоформы
    """
    try:
        # Делаем ресемплинг до нужной частоты
        if original_sample_rate != SAMPLE_RATE:
            resampler: torchaudio.transforms.Resample = torchaudio.transforms.Resample(original_sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Преобразуем в моно, если нужно
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Нормализация
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)
        
        return waveform
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "preprocess_audio",
            "Ошибка при извлечении аудиоформы"
        )
        return None


        
def compute_delta_features(mfcc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет дельта и дельта-дельта коэффициенты
    
    Args:
        mfcc: Тензор MFCC признаков
    
    Returns:
        Кортеж из дельта и дельта-дельта коэффициентов
    """
    try:
        delta: torch.Tensor = torchaudio.functional.compute_deltas(mfcc)
        delta2: torch.Tensor = torchaudio.functional.compute_deltas(delta)
        return delta, delta2
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "compute_delta_features",
            "Ошибка при вычислении дельта признаков"
        )
        return ()
    

def combine_features(mfcc: torch.Tensor, delta: torch.Tensor, delta2: torch.Tensor, 
                     spec_features: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Объединяет все признаки и приводит их к заданной длине
    
    Args:
        mfcc: Тензор MFCC признаков
        delta: Тензор дельта коэффициентов
        delta2: Тензор дельта-дельта коэффициентов
        spec_features: Тензор спектральных признаков
        target_length: Целевая длина
    
    Returns:
        Объединенный тензор признаков
    """
    try:
        combined_features: torch.Tensor = torch.cat([mfcc, delta, delta2, spec_features], dim=1)
        
        # Делаем pad или обрезаем до фиксированной длины
        if combined_features.size(2) < target_length:
            pad: torch.Tensor = torch.zeros(1, combined_features.size(1), target_length - combined_features.size(2))
            combined_features = torch.cat([combined_features, pad], dim=2)
        else:
            combined_features = combined_features[:, :, :target_length]
        
        return combined_features.squeeze(0).transpose(0, 1)
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "combine_features",
            "Ошибка при объединении признаков"
        )
        return None
    

def extract_mfcc_features(fragment: torch.Tensor) -> torch.Tensor:
    """
    Извлекает MFCC признаки из фрагмента аудио
    
    Args:
        fragment: Фрагмент аудиоформы
    
    Returns:
        Тензор MFCC признаков
    """
    try:
        mfcc_transform: torchaudio.transforms.MFCC = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=40,
            log_mels=True,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
        )
        return mfcc_transform(fragment)
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "extract_mfcc_features",
            "Ошибка при выделении mfcc"
        )
        return None
    

def apply_noise_reduction(waveform: torch.Tensor) -> torch.Tensor:
    """
    Применяет шумоподавление к аудиоформе
    
    Args:
        waveform: Тензор аудиоформы
    
    Returns:
        Аудиоформа с подавленным шумом
    """
    try:
        # 1. Вычисляем спектрограмму с окном Хэмминга
        window_fn = torch.hamming_window
        spec: torch.Tensor = torchaudio.transforms.Spectrogram(
            n_fft=1024, 
            hop_length=512,
            window_fn=window_fn,
            power=2
        )(waveform)
        
        # 2. Оценка шума из нескольких самых тихих фреймов
        frame_energies = torch.sum(spec, dim=1)
        num_noise_frames = min(20, spec.size(2) // 4)
        _, frame_indices = torch.topk(frame_energies, num_noise_frames, largest=False, dim=1)
        noise_frames = torch.stack([spec[:, :, i] for i in frame_indices[0]], dim=2)
        noise_estimate: torch.Tensor = torch.mean(noise_frames, dim=2, keepdim=True)
        
        # 3. Спектральное вычитание с мягким порогом и сохранением фазы
        enhanced_spec: torch.Tensor = torch.maximum(spec - 2 * noise_estimate, spec * 0.1)
        
        # 4. Обратное преобразование в волновую форму с более плавным синтезом
        griffin_lim: torchaudio.transforms.GriffinLim = torchaudio.transforms.GriffinLim(
            n_fft=1024, 
            hop_length=512,
            window_fn=window_fn,
            power=2,
            n_iter=32
        )
        enhanced_waveform: torch.Tensor = griffin_lim(enhanced_spec)
        
        return enhanced_waveform
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "apply_noise_reduction",
            "Ошибка при шумоподвалении"
        )
        return None
    

def remove_silence(waveform: torch.Tensor) -> torch.Tensor:
    """
    Удаляет тишину из аудиоформы
    
    Args:
        waveform: Тензор аудиоформы
    
    Returns:
        Аудиоформа без тишины
    """
    try:
        silence_threshold = 0.01
        is_silence = torch.abs(waveform) < silence_threshold
        non_silence_indices = torch.where(~is_silence)[1]
        
        if len(non_silence_indices) > 0:
            start_idx = max(0, non_silence_indices[0] - int(0.1 * SAMPLE_RATE))  # Добавляем 100мс до начала
            end_idx = min(waveform.size(1), non_silence_indices[-1] + int(0.1 * SAMPLE_RATE))  # Добавляем 100мс после конца
            waveform = waveform[:, start_idx:end_idx]
            
            # Повторная нормализация после удаления тишины
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)
        
        return waveform
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "remove_silence",
            "Ошибка при удалении тишины"
        )
        return None
    

def extract_spectral_features(fragment: torch.Tensor, target_time_dim: int, spectral_bands: int = 40) -> torch.Tensor:
    """
    Извлекает спектральные признаки из фрагмента аудио
    
    Args:
        fragment: Фрагмент аудиоформы
        target_time_dim: Целевая временная размерность
        spectral_bands: Количество спектральных полос
    
    Returns:
        Тензор спектральных признаков
    """
    try:
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=2048,
            hop_length=512
        )(fragment)
        
        # Уменьшаем размерность спектральных признаков до соответствия MFCC
        if spectrogram.shape[2] > target_time_dim:
            # Уменьшаем количество временных кадров
            indices = torch.linspace(0, spectrogram.shape[2] - 1, target_time_dim).long()
            spectrogram = spectrogram[:, :, indices]
        
        # Сжимаем размерность спектрограммы для соответствия MFCC
        # Используем усреднение по частотным диапазонам
        freq_indices = torch.linspace(0, spectrogram.shape[1] - 1, spectral_bands).long()
        spec_features = spectrogram[:, freq_indices, :]
        
        return spec_features
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "extract_spectral_features",
            "Ошибка при извлечении спектральных признаков"
        )
        return []
    

def split_into_fragments(waveform: torch.Tensor) -> List[torch.Tensor]:
    """
    Разбивает аудиоформу на фрагменты с перекрытием
    
    Args:
        waveform: Тензор аудиоформы
    
    Returns:
        Список фрагментов аудиоформы
    """
    try:
        fragments = []
        fragment_length: int = int(SAMPLE_RATE * AUDIO_FRAGMENT_LENGTH)
        hop_length: int = fragment_length // 2  # 50% перекрытие для увеличения числа фрагментов
        
        # Количество фрагментов с учетом перекрытия
        num_fragments: int = max(1, 1 + (waveform.size(1) - fragment_length) // hop_length)
        
        for i in range(num_fragments):
            start: int = i * hop_length
            end: int = min(start + fragment_length, waveform.size(1))
            
            fragment: torch.Tensor = waveform[:, start:end]
            
            # Если фрагмент слишком короткий, дополняем его нулями
            if end - start < fragment_length:
                padding: torch.Tensor = torch.zeros(1, fragment_length - (end - start))
                fragment = torch.cat([fragment, padding], dim=1)
            
            fragments.append(fragment)
        
        return fragments
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_to_features",
            "split_into_fragments",
            "Ошибка при сплите"
        )
        return []