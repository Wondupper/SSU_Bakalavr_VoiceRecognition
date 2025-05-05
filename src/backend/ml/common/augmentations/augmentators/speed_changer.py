import torch
import torchaudio
from typing import List
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE, AUGMENTATION

def change_speed(waveform: torch.Tensor) -> torch.Tensor:
    try:
        new_augmented_waveforms: List[torch.Tensor] = []
        for speed in AUGMENTATION['SPEEDS']:
            effects: List[List[str]] = [
                ["speed", str(speed)],
                ["rate", str(SAMPLE_RATE)]
            ]
            aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, SAMPLE_RATE, effects)
            new_augmented_waveforms.append(aug_waveform)
        return new_augmented_waveforms
        
    except Exception as e: 
        error_logger.log_exception(
            e,
            "speed_changer",
            "change_speed",
            "Ошибка при изменении скорости аудио"
        )
        # В случае ошибки возвращаем только оригинальное аудио
        return []