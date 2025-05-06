import os
import io
from typing import Dict, List, Tuple, Any, Optional
from werkzeug.datastructures import FileStorage
from src.backend.config import DATA_EMOTIONS, DATA_VOICE
from src.backend.loggers.error_logger import error_logger
from src.backend.loggers.info_logger import info_logger

def load_emotions_dataset() -> Dict[str, FileStorage]:
    """
    Загружает набор данных для обучения модели распознавания эмоций из конфигурации.
    
    Returns:
        Dict[str, FileStorage]: Словарь, где ключи - это эмоции, а значения аудиофайлы
    """
    dataset = {}
    module_name = "data_loader"
    
    try:
        info_logger.info(f"{module_name} - Начало загрузки набора данных для модели распознавания эмоций")
        
        # Проверяем наличие данных в конфигурации
        if not DATA_EMOTIONS or len(DATA_EMOTIONS) == 0:
            error_logger.log_error(
                "В конфигурации отсутствуют данные для обучения модели распознавания эмоций",
                module_name,
                "load_emotions_dataset"
            )
            return {}
        
        # Загружаем аудиофайлы для каждой эмоции
        for emotion, audio_path in DATA_EMOTIONS.items():
            # Проверяем существование файла
            if not os.path.exists(audio_path):
                error_logger.log_error(
                    f"Файл {audio_path} для эмоции {emotion} не найден",
                    module_name,
                    "load_emotions_dataset"
                )
                continue
            
            # Создаем объект FileStorage из файла на диске
            with open(audio_path, 'rb') as f:
                file_content = f.read()
            
            audio_file = FileStorage(
                stream=io.BytesIO(file_content),
                filename=os.path.basename(audio_path),
                content_type="audio/wav"
            )
            
            
            dataset[emotion] = audio_file
            
            info_logger.info(f"{module_name} - Файл {os.path.basename(audio_path)} для эмоции {emotion} успешно загружен")
                
        
        if not dataset:
            error_logger.log_error(
                "Не удалось загрузить ни один аудиофайл для обучения модели распознавания эмоций",
                module_name,
                "load_emotions_dataset"
            )
        else:
            info_logger.info(f"{module_name} - Набор данных для модели распознавания эмоций успешно загружен. "
                            f"Загружено {sum(len(files) for files in dataset.values())} файлов для {len(dataset)} эмоций")
        
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
    Загружает набор данных для обучения модели идентификации голоса из конфигурации.
    
    Returns:
        Dict[str, FileStorage]: Словарь, где ключи - это имена, а значения - аудиофайлы
    """
    dataset = {}
    module_name = "data_loader"
    
    try:
        info_logger.info(f"{module_name} - Начало загрузки набора данных для модели идентификации голоса")
        
        # Проверяем наличие данных в конфигурации
        if not DATA_VOICE or len(DATA_VOICE) == 0:
            error_logger.log_error(
                "В конфигурации отсутствуют данные для обучения модели идентификации голоса",
                module_name,
                "load_voice_dataset"
            )
            return {}
        
        # Загружаем аудиофайлы для каждого имени
        for name, audio_path in DATA_VOICE.items():
            # Проверяем существование файла
            if not os.path.exists(audio_path):
                error_logger.log_error(
                    f"Файл {audio_path} для имени {name} не найден",
                    module_name,
                    "load_voice_dataset"
                )
                continue
            
            # Создаем объект FileStorage из файла на диске
            with open(audio_path, 'rb') as f:
                file_content = f.read()
            
            audio_file = FileStorage(
                stream=io.BytesIO(file_content),
                filename=os.path.basename(audio_path),
                content_type="audio/wav"
            )
            
            
            dataset[name] = audio_file
            
            info_logger.info(f"{module_name} - Файл {os.path.basename(audio_path)} для имени {name} успешно загружен")
                
        
        if not dataset:
            error_logger.log_error(
                "Не удалось загрузить ни один аудиофайл для обучения модели идентификации голоса",
                module_name,
                "load_voice_dataset"
            )
        else:
            info_logger.info(f"{module_name} - Набор данных для модели идентификации голоса успешно загружен. "
                            f"Загружено {sum(len(files) for files in dataset.values())} файлов для {len(dataset)} говорящих")
        
        return dataset
    
    except Exception as e:
        error_logger.log_exception(
            e,
            module_name,
            "load_voice_dataset",
            "Ошибка при загрузке набора данных для модели идентификации голоса"
        )
        return {}
