import torch
import torchaudio
import random
from typing import List
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE, AUGMENTATION

def add_masking(waveform: torch.Tensor) -> torch.Tensor:
    try:
        new_augmented_waveforms: List[torch.Tensor] = []
        # 4. Маскирование по времени (Time Masking)
        for mask_param in AUGMENTATION['MASK_PARAMS']:
            mask_waveform: torch.Tensor = waveform.clone()
            time_mask_samples: int = int(mask_param * waveform.size(1))
            if time_mask_samples > 0:
                mask_start: int = random.randint(0, waveform.size(1) - time_mask_samples)
                mask_waveform[:, mask_start:mask_start + time_mask_samples] = 0
                new_augmented_waveforms.append(mask_waveform)
        return new_augmented_waveforms
    except Exception as e: 
        error_logger.log_exception(
            e,
            "time_maskiner",
            "add_masking",
            "Ошибка при добавлении маскирования в аудио"
        )
        # В случае ошибки возвращаем только оригинальное аудио
        return []
        