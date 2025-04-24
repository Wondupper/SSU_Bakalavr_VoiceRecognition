import io
from backend.api.error_logger import error_logger
from backend.processors.audio_processors.audio_processor import process_audio
from backend.processors.dataset_creators.dataset_creator import create_voice_id_dataset, create_emotion_dataset

def create_voice_id_dataset_from_audio(audio_file, name):
    """
    Создает датасет для модели идентификации по голосу из аудиофайла

    Args:
        audio_file: Файл-подобный объект с аудиозаписью (должен поддерживать метод read())
        name: Строка с именем пользователя для идентификации

    Returns:
        dataset: Список словарей с признаками и метками для обучения модели
                Каждый элемент содержит:
                - 'features': numpy массив с извлеченными признаками
                - 'label': строка с именем пользователя

    Raises:
        ValueError: Если входные данные некорректны или произошла ошибка обработки
    """
    try:
        # Проверка входных данных
        if not audio_file:
            raise ValueError("Не предоставлен аудиофайл")
        
        if not name or not isinstance(name, str) or len(name.strip()) == 0:
            raise ValueError("Не указано имя пользователя или оно некорректно")
        
        # Шаг 1: Обработка аудиофайла и получение аудиофрагментов
        audio_fragments = process_audio(audio_file)
        
        if not audio_fragments or len(audio_fragments) == 0:
            raise ValueError("Не удалось получить аудиофрагменты из файла")
        
        # Шаг 2: Создание датасета из аудиофрагментов
        # Аугментация происходит внутри create_voice_id_dataset
        dataset = create_voice_id_dataset(audio_fragments, name)
        
        return dataset
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "processors_main.py",
            "create_voice_id_dataset_from_audio",
            f"Ошибка при создании датасета для распознавания голоса пользователя {name}"
        )
        raise

def create_emotion_dataset_from_audio(audio_file, emotion):
    """
    Создает датасет для модели распознавания эмоций из аудиофайла

    Args:
        audio_file: Файл-подобный объект с аудиозаписью (должен поддерживать метод read())
        emotion: Строка с названием эмоции ('гнев', 'радость', 'грусть')

    Returns:
        dataset: Список словарей с признаками и метками для обучения модели
                Каждый элемент содержит:
                - 'features': numpy массив с извлеченными признаками
                - 'label': целое число - индекс эмоции (0 - гнев, 1 - радость, 2 - грусть)

    Raises:
        ValueError: Если входные данные некорректны или произошла ошибка обработки
    """
    try:
        # Проверка входных данных
        if not audio_file:
            raise ValueError("Не предоставлен аудиофайл")
        
        valid_emotions = ['гнев', 'радость', 'грусть']
        if not emotion or not isinstance(emotion, str) or emotion not in valid_emotions:
            raise ValueError(f"Некорректная эмоция. Должна быть одна из: {', '.join(valid_emotions)}")
        
        # Шаг 1: Обработка аудиофайла и получение аудиофрагментов
        audio_fragments = process_audio(audio_file)
        
        if not audio_fragments or len(audio_fragments) == 0:
            raise ValueError("Не удалось получить аудиофрагменты из файла")
        
        # Шаг 2: Создание датасета из аудиофрагментов
        # Аугментация происходит внутри create_emotion_dataset
        dataset = create_emotion_dataset(audio_fragments, emotion)
        
        return dataset
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "processors_main.py",
            "create_emotion_dataset_from_audio",
            f"Ошибка при создании датасета для распознавания эмоции {emotion}"
        )
        raise

# Функция для тестирования обработки из байтового потока
def process_audio_bytes_to_voice_id_dataset(audio_bytes, name):
    """
    Вспомогательная функция для создания датасета из байтов аудио
    
    Args:
        audio_bytes: Байты аудиофайла
        name: Строка с именем пользователя
        
    Returns:
        dataset: Датасет для обучения модели идентификации
    """
    audio_file = io.BytesIO(audio_bytes)
    return create_voice_id_dataset_from_audio(audio_file, name)

# Функция для тестирования обработки из байтового потока
def process_audio_bytes_to_emotion_dataset(audio_bytes, emotion):
    """
    Вспомогательная функция для создания датасета из байтов аудио
    
    Args:
        audio_bytes: Байты аудиофайла
        emotion: Строка с названием эмоции
        
    Returns:
        dataset: Датасет для обучения модели распознавания эмоций
    """
    audio_file = io.BytesIO(audio_bytes)
    return create_emotion_dataset_from_audio(audio_file, emotion)
