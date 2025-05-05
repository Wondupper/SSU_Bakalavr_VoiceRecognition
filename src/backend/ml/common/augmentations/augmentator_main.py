import torch
from typing import List
from src.backend.ml.common.augmentations.augmentators.noise_adder import add_noise
from src.backend.ml.common.augmentations.augmentators.speed_changer import change_speed
from src.backend.ml.common.augmentations.augmentators.reverberator import add_reverbiration
from src.backend.ml.common.augmentations.augmentators.time_maskiner import add_masking

def apply_augmentation(waveform: torch.Tensor) -> List[torch.Tensor]:
    """
    Применяет аугментацию к аудиофайлу для расширения обучающей выборки.
    
    Args:
        waveform: Тензор аудио [channels, time]
        module_name: Имя модуля для логирования
        
    Returns:
        Список аугментированных аудиофайлов
    """
    final_waveforms: List[torch.Tensor] = [waveform]  # Добавляем оригинальное аудио
    
    final_waveforms.extend(change_speed(waveform=waveform))
    final_waveforms.extend(add_reverbiration(waveform=waveform))
    final_waveforms.extend(add_masking(waveform=waveform))
    final_waveforms.extend(add_noise(waveform))
    
    return final_waveforms
