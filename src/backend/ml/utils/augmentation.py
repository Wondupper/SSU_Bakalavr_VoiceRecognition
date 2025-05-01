import torch
import torchaudio
import random
from typing import List
from backend.api.error_logger import error_logger
from backend.config import SAMPLE_RATE, AUGMENTATION

def apply_augmentation(waveform: torch.Tensor, module_name: str) -> List[torch.Tensor]:
    """
    Применяет аугментацию к аудиофайлу для расширения обучающей выборки.
    
    Args:
        waveform: Тензор аудио [channels, time]
        module_name: Имя модуля для логирования
        
    Returns:
        Список аугментированных аудиофайлов
    """
    
    try:
        augmented_waveforms: List[torch.Tensor] = [waveform]  # Добавляем оригинальное аудио
        
        new_augmented_waveforms: List[torch.Tensor] = []
        
        # 1. Ускорение аудио (time stretching)
        for speed in AUGMENTATION['FAST_SPEEDS']:
            effects: List[List[str]] = [
                ["speed", str(speed)],
                ["rate", str(SAMPLE_RATE)]
            ]
            aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, SAMPLE_RATE, effects)
            new_augmented_waveforms.append(aug_waveform)
        
        
        augmented_waveforms.extend(new_augmented_waveforms)
        new_augmented_waveforms = []
        
        
        # 2. Замедление аудио
        for speed in AUGMENTATION['SLOW_SPEEDS']:
            effects = [
                ["speed", str(speed)],
                ["rate", str(SAMPLE_RATE)]
            ]
            aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, SAMPLE_RATE, effects)
            new_augmented_waveforms.append(aug_waveform)
        

        augmented_waveforms.extend(new_augmented_waveforms)
        new_augmented_waveforms = []
        
        
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
        
        
        augmented_waveforms.extend(new_augmented_waveforms)
        new_augmented_waveforms = []
        
        
        # 4. Маскирование по времени (Time Masking)
        for mask_param in AUGMENTATION['MASK_PARAMS']:
            mask_waveform: torch.Tensor = waveform.clone()
            time_mask_samples: int = int(mask_param * waveform.size(1))
            if time_mask_samples > 0:
                mask_start: int = random.randint(0, waveform.size(1) - time_mask_samples)
                mask_waveform[:, mask_start:mask_start + time_mask_samples] = 0
                new_augmented_waveforms.append(mask_waveform)
        
        
        augmented_waveforms.extend(new_augmented_waveforms)
        new_augmented_waveforms = []
        
        
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
        
        
        augmented_waveforms.extend(new_augmented_waveforms)
            
        return augmented_waveforms
        
    except Exception as e:
        error_logger.log_exception(
            e,
            module_name,
            "apply_augmentation",
            "Ошибка при аугментации аудио"
        )
        # В случае ошибки возвращаем только оригинальное аудио
        return [waveform]
