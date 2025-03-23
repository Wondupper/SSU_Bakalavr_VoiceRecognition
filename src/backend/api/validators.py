from fastapi import HTTPException
from typing import List
from fastapi import UploadFile
import re
from ..config import AudioConfig

async def validate_training_data(files: List[UploadFile], user_name: str):
    """
    Валидация данных для обучения моделей
    """
    # Проверка количества файлов
    if len(files) > AudioConfig.MAX_AUDIOFILES_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Превышено максимальное количество файлов (максимум {AudioConfig.MAX_AUDIOFILES_COUNT})"
        )

    # Проверка общего размера файлов
    total_size = sum(file.size for file in files)
    if total_size > AudioConfig.MAX_AUDIOFILES_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Превышен максимальный общий размер файлов (максимум {AudioConfig.MAX_AUDIOFILES_SIZE // 1024 // 1024} МБ)"
        )

    # Проверка формата файлов
    for file in files:
        if not file.content_type == AudioConfig.ALLOWED_AUDIO_FORMAT:
            raise HTTPException(
                status_code=400,
                detail="Поддерживаются только WAV файлы"
            )

    # Проверка имени пользователя (только латинские буквы)
    if not re.match(r'^[a-zA-Z]+$', user_name):
        raise HTTPException(
            status_code=400,
            detail="Имя должно содержать только латинские буквы"
        )

async def validate_identification_data(file: UploadFile):
    """
    Валидация данных для идентификации
    """
    # Проверка размера файла
    if file.size > AudioConfig.INPUT_AUDIO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Размер файла не должен превышать {AudioConfig.INPUT_AUDIO_SIZE // 1024 // 1024} МБ"
        )

    # Проверка формата файла
    if not file.content_type == AudioConfig.ALLOWED_AUDIO_FORMAT:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только WAV файлы"
        ) 