import torch
import torchaudio
import random
from typing import List
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE, AUGMENTATION

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