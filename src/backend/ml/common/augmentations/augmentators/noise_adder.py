import torch
import torchaudio
import random
from typing import List
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE, AUGMENTATION

def add_noise(waveform: torch.Tensor) -> torch.Tensor:
    try:
        new_augmented_waveforms: List[torch.Tensor] = []
        # 5. Добавление шума
        for snr_db in AUGMENTATION['SNR_DBS']:
            noise: torch.Tensor = torch.randn_like(waveform)
            # Рассчитываем энергию сигнала и шума
            signal_power: torch.Tensor = torch.mean(waveform ** 2)
            noise_power: torch.Tensor = torch.mean(noise ** 2)
            # Корректируем шум для достижения нужного SNR
            snr: float = 10 ** (snr_db / 10)
            noise_scale: torch.Tensor = torch.sqrt(signal_power / (noise_power * snr))
            scaled_noise: torch.Tensor = noise * noise_scale
            # Добавляем шум к сигналу
            noisy_waveform: torch.Tensor = waveform + scaled_noise
            # Нормализация
            noisy_waveform = noisy_waveform / (torch.max(torch.abs(noisy_waveform)) + 1e-6)
            new_augmented_waveforms.append(noisy_waveform)
        return new_augmented_waveforms
    except Exception as e: 
        error_logger.log_exception(
            e,
            "noise_adder",
            "add_noise",
            "Ошибка при добавлении шума в аудио"
        )
        # В случае ошибки возвращаем только оригинальное аудио
        return [] 
        