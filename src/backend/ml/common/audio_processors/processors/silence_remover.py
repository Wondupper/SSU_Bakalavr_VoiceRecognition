import torch
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE

def remove_silence(waveform: torch.Tensor) -> torch.Tensor:
    """
    Удаляет тишину из аудиоформы
    
    Args:
        waveform: Тензор аудиоформы
    
    Returns:
        Аудиоформа без тишины
    """
    try:
        silence_threshold = 0.01
        is_silence = torch.abs(waveform) < silence_threshold
        non_silence_indices = torch.where(~is_silence)[1]
        
        if len(non_silence_indices) > 0:
            start_idx = max(0, non_silence_indices[0] - int(0.1 * SAMPLE_RATE))  # Добавляем 100мс до начала
            end_idx = min(waveform.size(1), non_silence_indices[-1] + int(0.1 * SAMPLE_RATE))  # Добавляем 100мс после конца
            waveform = waveform[:, start_idx:end_idx]
            
            # Повторная нормализация после удаления тишины
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)
        
        return waveform
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "silence_remover",
            "remove_silence",
            "Ошибка при удалении тишины"
        )
        return None