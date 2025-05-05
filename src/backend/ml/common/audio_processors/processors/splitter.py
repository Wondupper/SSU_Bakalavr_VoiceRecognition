import torch
from typing import List
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH

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
            "splitter",
            "split_into_fragments",
            "Ошибка при сплите"
        )
        return None

