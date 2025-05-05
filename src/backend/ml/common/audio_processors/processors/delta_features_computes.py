import torch
import torchaudio
from typing import List, Optional, Tuple
from src.backend.loggers.error_logger import error_logger

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
            "delta_features_computes",
            "compute_delta_features",
            "Ошибка при вычислении дельта признаков"
        )
        return ()