import torch
import torchaudio
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE

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
            "mfcc_extractor",
            "extract_mfcc_features",
            "Ошибка при выделении mfcc"
        )
        return None