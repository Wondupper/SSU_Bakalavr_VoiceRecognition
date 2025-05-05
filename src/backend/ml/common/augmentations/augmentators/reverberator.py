import torch
import torchaudio
import random
from typing import List
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE, AUGMENTATION

def add_reverbiration(waveform: torch.Tensor) -> torch.Tensor:
    try:
        new_augmented_waveforms: List[torch.Tensor] = []
        # 3. Реверберация (добавление эхо)
        for decay in AUGMENTATION['DECAYS']:
            reverb_waveform: torch.Tensor = waveform.clone()
            # Создаем простую реверберацию, добавляя задержанную и затухающую копию сигнала
            delay_samples: int = int(0.05 * SAMPLE_RATE)  # 50 мс задержка
            if waveform.size(1) > delay_samples:
                reverb: torch.Tensor = torch.zeros_like(waveform)
                reverb[:, delay_samples:] = waveform[:, :-delay_samples] * decay
                reverb_waveform = waveform + reverb
                # Нормализация
                reverb_waveform = reverb_waveform / (torch.max(torch.abs(reverb_waveform)) + 1e-6)
                new_augmented_waveforms.append(reverb_waveform)
        return new_augmented_waveforms
    except Exception as e: 
        error_logger.log_exception(
            e,
            "reverberator",
            "add_reverbiration",
            "Ошибка при добавлении ревербирации в аудио"
        )
        # В случае ошибки возвращаем только оригинальное аудио
        return []
        