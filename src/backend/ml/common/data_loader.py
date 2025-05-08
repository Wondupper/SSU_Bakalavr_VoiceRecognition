import os
import io
from typing import Dict, List, Tuple, Any, Optional
from werkzeug.datastructures import FileStorage
from backend.config import DATA_EMOTIONS, DATA_VOICE
from backend.loggers.error_logger import error_logger

def load_emotions_dataset() -> Dict[str, FileStorage]:
    """
    Загрузка набора данных для обучения модели распознавания эмоций из конфигурации.
    
    Returns:
        Dict[str, FileStorage]: Словарь, где ключи - это эмоции, а значения аудиофайлы
    """
    dataset = {}
    module_name = "data_loader"
    
    try:
        
        # Загружаем аудиофайлы для каждой эмоции
        for emotion, audio_path in DATA_EMOTIONS.items():
            
            # Создаем объект FileStorage из файла на диске
            with open(audio_path, 'rb') as f:
                file_content = f.read()
            
            audio_file = FileStorage(
                stream=io.BytesIO(file_content),
                filename=os.path.basename(audio_path),
                content_type="audio/wav"
            )
            
            
            dataset[emotion] = audio_file
            
        
        return dataset
    
    except Exception as e:
        error_logger.log_exception(
            e,
            module_name,
            "load_emotions_dataset",
            "Ошибка при загрузке набора данных для модели распознавания эмоций"
        )
        return {}

def load_voice_dataset() -> Dict[str, FileStorage]:
    """
    Загрузка набора данных для обучения модели идентификации голоса из конфигурации.
    
    Returns:
        Dict[str, FileStorage]: Словарь, где ключи - это имена, а значения - аудиофайлы
    """
    dataset = {}
    module_name = "data_loader"
    
    try:
        
        # Загружаем аудиофайлы для каждого имени
        for name, audio_path in DATA_VOICE.items():
            
            # Создаем объект FileStorage из файла на диске
            with open(audio_path, 'rb') as f:
                file_content = f.read()
            
            audio_file = FileStorage(
                stream=io.BytesIO(file_content),
                filename=os.path.basename(audio_path),
                content_type="audio/wav"
            )
            
            
            dataset[name] = audio_file
        
        return dataset
    
    except Exception as e:
        error_logger.log_exception(
            e,
            module_name,
            "load_voice_dataset",
            "Ошибка при загрузке набора данных для модели идентификации голоса"
        )
        return {}
