import torch
import torchaudio
from src.backend.loggers.error_logger import error_logger

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
            "spectral_features_extractor",
            "extract_spectral_features",
            "Ошибка при извлечении спектральных признаков"
        )
        return None