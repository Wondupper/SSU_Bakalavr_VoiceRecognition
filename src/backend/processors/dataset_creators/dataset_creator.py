from backend.processors.augmentation_processors.augmentation_processor import augment_audio
from backend.api.error_logger import error_logger
from .other.feature_extractors import extract_voice_id_features, extract_emotion_features
from .other.validation import validate_audio_fragments, validate_emotion, validate_dataset_result
from backend.config import DATASET_CREATOR

# Используем константы из конфигурационного файла
EMOTIONS = DATASET_CREATOR['EMOTIONS']

def create_voice_id_dataset(audio_fragments, name):
    """
    Создание датасета для модели идентификации по голосу

    Args:
        audio_fragments: Список аудиофрагментов
        name: Строка с именем пользователя

    Returns:
        dataset: Список словарей с признаками и метками для обучения модели
                Каждый элемент содержит:
                - 'features': numpy массив формы (MAX_FRAMES, n_features)
                - 'label': строка с именем пользователя
    """
    # Валидация входных данных
    validate_audio_fragments(audio_fragments, "voice_id")
    
    # Аугментация аудиофрагментов с ограничением результатов
    try:
        augmented_fragments = augment_audio(audio_fragments)
    except Exception as e:
        error_info = error_logger.log_exception(
            e,
            "dataset_creator",
            "augmentation",
            "Ошибка при аугментации. Продолжаем без аугментации."
        )
        # Если аугментация не удалась, используем только исходные фрагменты
        augmented_fragments = audio_fragments.copy()

    # Создание датасета для обучения нейронной сети
    dataset = []
    
    # Последовательное извлечение признаков из каждого фрагмента
    for fragment in augmented_fragments:
        try:
            features = extract_voice_id_features(fragment)
            if features is not None:
                dataset.append({'features': features, 'label': name})
        except Exception as e:
            error_info = error_logger.log_exception(
                e,
                "dataset_creator",
                "feature_extraction",
                "Ошибка при извлечении признаков из фрагмента"
            )
            # Пропускаем фрагмент в случае ошибки
            continue
    
    # Проверка наличия данных в датасете
    validate_dataset_result(dataset, "voice_id")
    
    return dataset

def create_emotion_dataset(audio_fragments, emotion):
    """
    Создание датасета для модели распознавания эмоций
    
    Args:
        audio_fragments: Список аудиофрагментов для обучения
        emotion: Строка с названием эмоции ('гнев', 'радость', 'грусть')
    
    Returns:
        dataset: Список словарей с признаками и метками для обучения модели
                Каждый элемент содержит:
                - 'features': numpy массив формы (MAX_FRAMES, n_features)
                - 'label': целое число - индекс эмоции (0 - гнев, 1 - радость, 2 - грусть)
    """
    # Валидация входных данных
    validate_audio_fragments(audio_fragments, "emotion")
    validate_emotion(emotion)
    
    # Аугментация аудиофрагментов с ограничением результатов
    try:
        augmented_fragments = augment_audio(audio_fragments)
    except Exception as e:
        error_info = error_logger.log_exception(
            e,
            "dataset_creator",
            "augmentation",
            "Ошибка при аугментации. Продолжаем без аугментации."
        )
        # Если аугментация не удалась, используем только исходные фрагменты
        augmented_fragments = audio_fragments.copy()
    
    # Создание датасета для обучения нейронной сети
    dataset = []
    
    # Эмоции кодируем как индексы
    emotion_map = {'гнев': 0, 'радость': 1, 'грусть': 2}
    emotion_idx = emotion_map[emotion]
    
    # Последовательное извлечение признаков из каждого фрагмента
    for fragment in augmented_fragments:
        try:
            features = extract_emotion_features(fragment)
            if features is not None:
                dataset.append({'features': features, 'label': emotion_idx})
        except Exception as e:
            error_info = error_logger.log_exception(
                e,
                "dataset_creator",
                "feature_extraction",
                "Ошибка при извлечении признаков из фрагмента для эмоций"
            )
            # Пропускаем фрагмент в случае ошибки
            continue
    
    # Проверка наличия данных в датасете
    validate_dataset_result(dataset, "emotion")
    
    return dataset
