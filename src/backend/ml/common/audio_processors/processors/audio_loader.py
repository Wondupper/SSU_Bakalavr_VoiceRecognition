import torch
import torchaudio
import io
from typing import List, Optional, Tuple
from werkzeug.datastructures import FileStorage
from src.backend.loggers.error_logger import error_logger

def load_audio_from_file(audio_file: FileStorage) -> Tuple[torch.Tensor, int]:
    """
    Загружает аудио из файла и возвращает wavform и sample_rate
    
    Args:
        audio_file: Файл аудио (объект FileStorage Flask)
    
    Returns:
        Tuple с waveform и sample_rate
    """
    try:
        # Сохраняем содержимое файла в буфер
        audio_buffer: io.BytesIO = io.BytesIO(audio_file.read())
        # Сбрасываем указатель в начало буфера
        audio_buffer.seek(0)
        # Загружаем аудио из буфера
        waveform: torch.Tensor
        sample_rate: int
        waveform, sample_rate = torchaudio.load(audio_buffer)
        # Сбрасываем указатель файла на начало для возможного дальнейшего использования
        audio_file.seek(0)
        
        return waveform, sample_rate
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_loader",
            "load_audio_from_file",
            "Ошибка при загрузке аудио из аудиофайла"
        )
        return ()