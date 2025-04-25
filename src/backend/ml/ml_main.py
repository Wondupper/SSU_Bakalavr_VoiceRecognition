import os
import time
import random
from backend.ml.voice_identification.model import VoiceIdentificationModel
from backend.ml.emotion_recognition.model import EmotionRecognitionModel
from backend.api.error_logger import error_logger
from backend.config import EMOTIONS, VOICE_ID_MODELS_DIR, EMOTION_MODELS_DIR
from backend.ml.shared.model_loader_or_saver import save_model as saver_save_model
from backend.ml.shared.model_loader_or_saver import load_model as saver_load_model
from backend.ml.shared.model_loader_or_saver import load_models as saver_load_models
from backend.ml.shared.model_loader_or_saver import save_models as saver_save_models

# Инициализация моделей
voice_id_model = VoiceIdentificationModel()
emotion_model = EmotionRecognitionModel()

# Переменные для отслеживания прогресса обучения
training_progress = {
    'voice_id': {
        'current_epoch': 0,
        'total_epochs': 0,
        'accuracy': 0.0,
        'loss': 0.0,
        'start_time': 0,
        'status': 'idle'  # 'idle', 'training', 'completed', 'error'
    },
    'emotion': {
        'current_epoch': 0,
        'total_epochs': 0,
        'accuracy': 0.0,
        'loss': 0.0,
        'start_time': 0,
        'status': 'idle'  # 'idle', 'training', 'completed', 'error'
    }
}

# Эмоция дня, которая генерируется один раз при запуске сервера
DAILY_EMOTION = random.choice(EMOTIONS)

def get_daily_emotion():
    """
    Возвращает эмоцию дня, которая остается постоянной до перезапуска сервера
    
    Returns:
        str: Эмоция дня
    """
    return DAILY_EMOTION

def get_training_status():
    """
    Получает текущий статус обучения моделей
    
    Returns:
        dict: Словарь с информацией о статусе обучения
    """
    return {
        'voice_id_training': not voice_id_model.is_training,
        'emotion_training': not emotion_model.is_training
    }

def get_training_progress(model_type='all'):
    """
    Получает прогресс обучения моделей
    
    Args:
        model_type: Тип модели ('voice_id', 'emotion' или 'all')
        
    Returns:
        dict: Прогресс обучения для запрошенного типа модели
    """
    if model_type == 'all':
        return training_progress
    elif model_type in ['voice_id', 'emotion']:
        return training_progress[model_type]
    else:
        error_logger.log_error(
            f"Неверный тип модели: {model_type}",
            "ml_main",
            "get_training_progress"
        )
        return None

def train_voice_id_model(dataset):
    """
    Обучает модель идентификации голоса.
    
    Args:
        dataset: Датасет для обучения модели.
        
    Returns:
        bool: успешность обучения модели
    """
    try:
        # Проверяем, не обучается ли модель в данный момент
        if voice_id_model.is_training:
            error_logger.log_error(
                "Модель уже обучается. Попробуйте позже",
                "ml_main",
                "train_voice_id_model"
            )
            return False
            
        # Обновляем статус обучения
        training_progress['voice_id']['status'] = 'training'
        training_progress['voice_id']['start_time'] = time.time()
        
        # Извлекаем аудиофрагменты и метки из датасета
        audio_fragments = [item['audio'] for item in dataset]
        labels = [item['label'] for item in dataset]
        
        # Обучаем модель
        result = voice_id_model.train(audio_fragments, labels)
        
        # Обновляем статус обучения
        if result:
            training_progress['voice_id']['status'] = 'completed'
        else:
            training_progress['voice_id']['status'] = 'error'
            
        return result
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "train_voice_id_model",
            "Ошибка при обучении модели идентификации голоса"
        )
        
        # Обновляем статус обучения при ошибке
        training_progress['voice_id']['status'] = 'error'
        return False
    
def train_emotion_model(dataset):
    """
    Обучает модель распознавания эмоций.
    
    Args:
        dataset: Датасет для обучения модели.
        
    Returns:
        bool: успешность обучения модели
    """
    try:
        # Проверяем, не обучается ли модель в данный момент
        if emotion_model.is_training:
            error_logger.log_error(
                "Модель уже обучается. Попробуйте позже",
                "ml_main",
                "train_emotion_model"
            )
            return False
            
        # Обновляем статус обучения
        training_progress['emotion']['status'] = 'training'
        training_progress['emotion']['start_time'] = time.time()
        
        # Извлекаем аудиофрагменты и метки из датасета
        audio_fragments = [item['audio'] for item in dataset]
        labels = [item['label'] for item in dataset]
        
        # Обучаем модель
        result = emotion_model.train(audio_fragments, labels)
        
        # Обновляем статус обучения
        if result:
            training_progress['emotion']['status'] = 'completed'
        else:
            training_progress['emotion']['status'] = 'error'
            
        return result
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "train_emotion_model",
            "Ошибка при обучении модели распознавания эмоций"
        )
        
        # Обновляем статус обучения при ошибке
        training_progress['emotion']['status'] = 'error'
        return False

def predict_user(audio_fragments):
    """
    Идентифицирует пользователя по голосу
    
    Args:
        audio_fragments: Список аудиофрагментов для идентификации
        
    Returns:
        str: Имя пользователя или "unknown", если не удалось идентифицировать
    """
    try:
        # Проверка состояния модели
        if not voice_id_model.is_trained:
            error_logger.log_error(
                "Модель идентификации не обучена",
                "ml_main",
                "predict_user"
            )
            return "unknown"
            
        # Идентификация пользователя
        results = voice_id_model.predict(audio_fragments)
        
        # Для идентификации голоса: голосование по всем фрагментам
        if not results:
            return "unknown"
        
        # Голосование по результатам фрагментов
        vote_counts = {}
        confidence_sums = {}
        
        for result in results:
            label = result['label']
            confidence = result['confidence']
            
            if label not in vote_counts:
                vote_counts[label] = 0
                confidence_sums[label] = 0
            
            vote_counts[label] += 1
            confidence_sums[label] += confidence
        
        # Находим метку с наибольшим количеством голосов
        if vote_counts:
            identity = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            avg_confidence = confidence_sums[identity] / vote_counts[identity]
            
            # Если средняя уверенность низкая или это "Unknown", считаем неизвестным
            if avg_confidence < 0.6 or identity == "Unknown":
                return "unknown"
                
            return identity
        else:
            return "unknown"
            
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "predict_user",
            "Ошибка при идентификации пользователя"
        )
        return "unknown"

def predict_emotion(audio_fragments):
    """
    Распознает эмоцию из аудиофрагментов
    
    Args:
        audio_fragments: Список аудиофрагментов для распознавания
        
    Returns:
        str: Распознанная эмоция или "unknown", если не удалось распознать
    """
    try:
        # Проверка состояния модели
        if not emotion_model.is_trained:
            error_logger.log_error(
                "Модель распознавания эмоций не обучена",
                "ml_main",
                "predict_emotion"
            )
            return "unknown"
            
        # Распознавание эмоции
        results = emotion_model.predict(audio_fragments)
        
        # Голосование по результатам фрагментов для эмоций
        vote_counts = {}
        confidence_sums = {}
        
        for result in results:
            label = result['label']
            confidence = result['confidence']
            
            if label not in vote_counts:
                vote_counts[label] = 0
                confidence_sums[label] = 0
            
            vote_counts[label] += 1
            confidence_sums[label] += confidence
        
        # Находим эмоцию с наибольшим количеством голосов
        if vote_counts:
            detected_emotion = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            avg_confidence = confidence_sums[detected_emotion] / vote_counts[detected_emotion]
            
            # Если средняя уверенность низкая, считаем неизвестной
            if avg_confidence < 0.6:
                return "unknown"
                
            return detected_emotion
        else:
            return "unknown"
            
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "predict_emotion",
            "Ошибка при распознавании эмоции"
        )
        return "unknown"

def identify_with_emotion(audio_fragments, expected_emotion):
    """
    Комплексная идентификация пользователя с проверкой эмоции
    
    Args:
        audio_fragments: Список аудиофрагментов для идентификации
        expected_emotion: Ожидаемая эмоция
        
    Returns:
        dict: Результаты идентификации и распознавания эмоции
    """
    # Блокировка доступа к моделям во время идентификации
    voice_id_model.is_training = True
    emotion_model.is_training = True
    
    try:
        # Проверка, что модели обучены
        if not voice_id_model.is_trained:
            return {
                'success': False,
                'message': 'Модель идентификации не обучена',
                'identity': None,
                'emotion': None,
                'match': False
            }
            
        if not emotion_model.is_trained:
            return {
                'success': False,
                'message': 'Модель эмоций не обучена',
                'identity': None,
                'emotion': None,
                'match': False
            }
        
        # Идентификация пользователя
        identity = predict_user(audio_fragments)
        
        # Распознавание эмоции
        detected_emotion = predict_emotion(audio_fragments)
        
        # Проверка совпадения эмоции с ожидаемой
        emotion_match = detected_emotion.lower() == expected_emotion.lower()
        
        # Рассчитываем успешность идентификации
        success = True
        message = "Идентификация выполнена успешно"
        
        if identity == "unknown" and detected_emotion == "unknown":
            success = False
            message = "Не удалось распознать пользователя и эмоцию"
        elif identity == "unknown":
            success = False
            message = "Не удалось распознать пользователя"
        elif detected_emotion == "unknown":
            success = False
            message = "Не удалось распознать эмоцию"
        elif not emotion_match:
            success = False
            message = f"Эмоция не соответствует ожидаемой ({detected_emotion} вместо {expected_emotion})"
        
        return {
            'success': success,
            'message': message,
            'identity': identity,
            'emotion': detected_emotion,
            'match': emotion_match
        }
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "identify_with_emotion",
            "Ошибка при идентификации пользователя с эмоцией"
        )
        return {
            'success': False,
            'message': f'Внутренняя ошибка: {str(e)}',
            'identity': None,
            'emotion': None,
            'match': False
        }
    finally:
        voice_id_model.is_training = False
        emotion_model.is_training = False

# Новые функции для каждого типа модели вместо функций с параметром model_type

def reset_voice_id_model():
    """
    Сброс модели идентификации голоса до начального состояния
    
    Returns:
        bool: Успешно ли сброшена модель
    """
    try:
        voice_id_model.reset_model()
        return True
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "reset_voice_id_model",
            "Ошибка при сбросе модели идентификации голоса"
        )
        return False

def reset_emotion_model():
    """
    Сброс модели распознавания эмоций до начального состояния
    
    Returns:
        bool: Успешно ли сброшена модель
    """
    try:
        emotion_model.reset_model()
        return True
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "reset_emotion_model",
            "Ошибка при сбросе модели распознавания эмоций"
        )
        return False

def load_voice_id_model(filepath):
    """
    Загрузка модели идентификации голоса из файла
    
    Args:
        filepath: Путь к файлу модели
        
    Returns:
        bool: Успешно ли загружена модель
    """
    try:
        # Загружаем модель с помощью общей функции
        model, is_trained, success = saver_load_model(filepath)
        
        if not success:
            return False
            
        # Применяем полученные данные к модели
        voice_id_model.model = model
        voice_id_model.is_trained = is_trained
        return True
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "load_voice_id_model",
            "Ошибка при загрузке модели идентификации голоса"
        )
        return False

def load_emotion_model(filepath):
    """
    Загрузка модели распознавания эмоций из файла
    
    Args:
        filepath: Путь к файлу модели
        
    Returns:
        bool: Успешно ли загружена модель
    """
    try:
        # Загружаем модель с помощью общей функции
        model, is_trained, success = saver_load_model(filepath)
        
        if not success:
            return False
            
        # Применяем полученные данные к модели
        emotion_model.model = model
        emotion_model.is_trained = is_trained
        return True
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "load_emotion_model",
            "Ошибка при загрузке модели распознавания эмоций"
        )
        return False

def save_voice_id_model():
    """
    Сохранение модели идентификации голоса в файл
    
    Returns:
        str или None: Путь к сохраненной модели или None в случае ошибки
    """
    try:
        # Базовая директория для сохранения
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        if voice_id_model.is_training:
            error_logger.log_error(
                "Модель используется. Попробуйте позже",
                "ml_main",
                "save_voice_id_model"
            )
            return None
        
        # Создаем директорию для сохранения
        save_dir = os.path.join(base_dir, VOICE_ID_MODELS_DIR)
        os.makedirs(save_dir, exist_ok=True)
        
        # Генерируем имя файла с текущим временем
        timestamp = int(time.time())
        filepath = os.path.join(save_dir, f"voice_id_model_{timestamp}")
        
        # Сохраняем модель с помощью общей функции
        if saver_save_model(voice_id_model.model, voice_id_model.is_trained, filepath):
            return filepath
        return None
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "save_voice_id_model",
            "Ошибка при сохранении модели идентификации голоса"
        )
        return None

def save_emotion_model():
    """
    Сохранение модели распознавания эмоций в файл
    
    Returns:
        str или None: Путь к сохраненной модели или None в случае ошибки
    """
    try:
        # Базовая директория для сохранения
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        if emotion_model.is_training:
            error_logger.log_error(
                "Модель используется. Попробуйте позже",
                "ml_main",
                "save_emotion_model"
            )
            return None
        
        # Создаем директорию для сохранения
        save_dir = os.path.join(base_dir, EMOTION_MODELS_DIR)
        os.makedirs(save_dir, exist_ok=True)
        
        # Генерируем имя файла с текущим временем
        timestamp = int(time.time())
        filepath = os.path.join(save_dir, f"emotion_model_{timestamp}")
        
        # Сохраняем модель с помощью общей функции
        if saver_save_model(emotion_model.model, emotion_model.is_trained, filepath):
            return filepath
        return None
    except Exception as e:
        error_logger.log_exception(
            e,
            "ml_main",
            "save_emotion_model",
            "Ошибка при сохранении модели распознавания эмоций"
        )
        return None

# Функции-обёртки для взаимодействия с main.py

def load_models(basedir):
    """
    Загрузка моделей при старте приложения
    
    Args:
        basedir: Базовая директория проекта
        
    Returns:
        tuple: (voice_id_loaded, emotion_loaded)
            voice_id_loaded: Была ли загружена модель идентификации
            emotion_loaded: Была ли загружена модель эмоций
    """
    return saver_load_models(voice_id_model, emotion_model, basedir)

def save_models(basedir):
    """
    Сохранение моделей при завершении работы приложения
    
    Args:
        basedir: Базовая директория проекта
        
    Returns:
        tuple: (voice_id_path, emotion_path)
            voice_id_path: Путь к сохраненной модели идентификации или None
            emotion_path: Путь к сохраненной модели эмоций или None
    """
    return saver_save_models(voice_id_model, emotion_model, basedir)
