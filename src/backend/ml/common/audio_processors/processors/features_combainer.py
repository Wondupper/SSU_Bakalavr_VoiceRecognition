import torch
from src.backend.loggers.error_logger import error_logger

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
            "features_combainer",
            "combine_features",
            "Ошибка при объединении признаков"
        )
        return None