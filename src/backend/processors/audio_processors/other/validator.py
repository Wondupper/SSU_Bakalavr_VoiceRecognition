import os
import numpy as np
from backend.api.error_logger import error_logger

def validate_audio_file(audio_file):
    """
    Проверка валидности аудиофайла
    """
    if not audio_file:
        error_logger.log_exception(
            ValueError("Аудиофайл не предоставлен"),
            "audio_processing",
            "validation",
            "Проверка входных данных"
        )
        raise ValueError("Аудиофайл не предоставлен")
    
    # Проверка формата аудиофайла
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    if file_extension not in ['.wav', '.mp3', '.ogg', '.flac']:
        error_logger.log_exception(
            ValueError(f"Неподдерживаемый формат аудиофайла: {file_extension}"),
            "audio_processing",
            "validation",
            "Проверка формата файла"
        )
        raise ValueError(f"Неподдерживаемый формат аудиофайла: {file_extension}")

def validate_audio_bytes(audio_bytes):
    """
    Проверка байтов аудиофайла
    """
    if len(audio_bytes) == 0:
        error_logger.log_exception(
            ValueError("Пустой аудиофайл"),
            "audio_processing",
            "validation",
            "Проверка размера файла"
        )
        raise ValueError("Пустой аудиофайл")

def validate_audio_data(audio_data, sr, min_length=None):
    """
    Проверка аудиоданных после загрузки
    """
    # Проверка на пустое аудио
    if len(audio_data) == 0 or np.all(audio_data == 0):
        error_logger.log_exception(
            ValueError("Аудиофайл не содержит данных"),
            "audio_processing",
            "validation",
            "Проверка содержимого аудио"
        )
        raise ValueError("Аудиофайл не содержит данных")
    
    # Проверка на недопустимые значения
    if np.isnan(audio_data).any() or np.isinf(audio_data).any():
        error_logger.log_error(
            "Обнаружены недопустимые значения в аудиоданных",
            "audio_processing",
            "validation"
        )
        audio_data = np.nan_to_num(audio_data)
    
    # Проверка минимальной длительности, если указана
    if min_length and len(audio_data) < min_length:
        error_logger.log_exception(
            ValueError(f"Аудио слишком короткое для обработки: {len(audio_data) / sr:.2f} секунд"),
            "audio_processing",
            "validation",
            "Проверка длительности аудио"
        )
        raise ValueError(f"Аудио слишком короткое для обработки: {len(audio_data) / sr:.2f} секунд")
    
    return audio_data 