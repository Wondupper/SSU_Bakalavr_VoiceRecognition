from backend.processors.augmentation_processors.augmentation_processor import augment_audio
from backend.api.error_logger import error_logger
from .other.feature_extractors import extract_voice_id_features, extract_emotion_features
from .other.validator import validate_audio_fragments, validate_emotion, validate_dataset_result


def create_voice_id_training_dataset(audio_fragments, name):
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

def create_emotion_training_dataset(audio_fragments, emotion):
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
    
    # Последовательное извлечение признаков из каждого фрагмента
    for fragment in augmented_fragments:
        try:
            features = extract_emotion_features(fragment)
            if features is not None:
                dataset.append({'features': features, 'label': emotion})
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

def create_voice_id_prediction_features(audio_fragments):
    """
    Создание набора признаков из аудиофрагментов для предсказания идентификации по голосу
    
    Args:
        audio_fragments: Список аудиофрагментов для анализа
    
    Returns:
        features_list: Список numpy массивов формы (MAX_FRAMES, n_features) с признаками для модели
    """
    # Валидация входных данных
    validate_audio_fragments(audio_fragments, "voice_id_prediction")
    
    # Создание списка признаков для предсказания
    features_list = []
    
    # Последовательное извлечение признаков из каждого фрагмента без аугментации
    for fragment in audio_fragments:
        try:
            features = extract_voice_id_features(fragment)
            if features is not None:
                features_list.append(features)
        except Exception as e:
            error_info = error_logger.log_exception(
                e,
                "dataset_creator",
                "create_voice_id_prediction_features",
                "Ошибка при извлечении признаков из фрагмента для предсказания"
            )
            # Пропускаем фрагмент в случае ошибки
            continue
    
    # Проверка наличия данных в списке признаков
    if not features_list:
        error_msg = "Не удалось создать признаки для предсказания: все операции извлечения признаков завершились с ошибками"
        error_info = error_logger.log_exception(
            ValueError(error_msg),
            "dataset_creator.py",
            "create_voice_id_prediction_features",
            "Проверка результатов обработки данных"
        )
        raise ValueError(error_msg)
    
    return features_list

def create_emotion_prediction_features(audio_fragments):
    """
    Создание набора признаков из аудиофрагментов для предсказания эмоции
    
    Args:
        audio_fragments: Список аудиофрагментов для анализа
    
    Returns:
        features_list: Список numpy массивов формы (MAX_FRAMES, n_features) с признаками для модели
    """
    # Валидация входных данных
    validate_audio_fragments(audio_fragments, "emotion_prediction")
    
    # Создание списка признаков для предсказания
    features_list = []
    
    # Последовательное извлечение признаков из каждого фрагмента без аугментации
    for fragment in audio_fragments:
        try:
            features = extract_emotion_features(fragment)
            if features is not None:
                features_list.append(features)
        except Exception as e:
            error_info = error_logger.log_exception(
                e,
                "dataset_creator",
                "create_emotion_prediction_features",
                "Ошибка при извлечении признаков из фрагмента для предсказания эмоции"
            )
            # Пропускаем фрагмент в случае ошибки
            continue
    
    # Проверка наличия данных в списке признаков
    if not features_list:
        error_msg = "Не удалось создать признаки для предсказания эмоции: все операции извлечения признаков завершились с ошибками"
        error_info = error_logger.log_exception(
            ValueError(error_msg),
            "dataset_creator.py",
            "create_emotion_prediction_features",
            "Проверка результатов обработки данных"
        )
        raise ValueError(error_msg)
    
    return features_list
