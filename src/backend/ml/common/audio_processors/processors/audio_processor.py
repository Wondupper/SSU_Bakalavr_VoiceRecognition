import torch
import torchaudio
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE

def preprocess_audio(waveform: torch.Tensor, original_sample_rate: int) -> torch.Tensor:
    """
    Выполняет предварительную обработку аудио: ресемплинг, преобразование в моно, нормализация
    
    Args:
        waveform: Тензор аудиоформы
        original_sample_rate: Исходная частота дискретизации
    
    Returns:
        Обработанный тензор аудиоформы
    """
    try:
        # Делаем ресемплинг до нужной частоты
        if original_sample_rate != SAMPLE_RATE:
            resampler: torchaudio.transforms.Resample = torchaudio.transforms.Resample(original_sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Преобразуем в моно, если нужно
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Нормализация
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)
        
        return waveform
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_processor",
            "preprocess_audio",
            "Ошибка при извлечении аудиоформы"
        )