import os
import time
import random
from backend.ml.voice_identification.model import VoiceIdentificationModel
from backend.ml.emotion_recognition.model import EmotionRecognitionModel
from backend.api.error_logger import error_logger
from backend.config import EMOTIONS

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
        'voice_id_training': voice_id_model.is_training,
        'emotion_training': emotion_model.is_training
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
        
        # Извлекаем признаки и метки из датасета
        features = [item['features'] for item in dataset]
        labels = [item['label'] for item in dataset]
        
        # Обучаем модель
        result = voice_id_model.train(features, labels)
        
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
        # Убедимся, что флаг обучения сброшен в случае ошибки
        voice_id_model.is_training = False
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
        
        # Извлекаем признаки и метки из датасета
        features = [item['features'] for item in dataset]
        labels = [item['label'] for item in dataset]
        
        # Обучаем модель
        result = emotion_model.train(features, labels)
        
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
        # Убедимся, что флаг обучения сброшен в случае ошибки
        emotion_model.is_training = False
        return False

def predict_user(features_list_id):
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
        results = voice_id_model.predict(features_list_id)
        
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

def predict_emotion(features_list_emotion):
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
        results = emotion_model.predict(features_list_emotion)
        
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
            if avg_confidence < 0.8:
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

def identify_with_emotion(features_list_id, features_list_emotion, expected_emotion):
    """
    Комплексная идентификация пользователя с проверкой эмоции
    
    Args:
        audio_fragments: Список аудиофрагментов для идентификации
        expected_emotion: Ожидаемая эмоция
        
    Returns:
        dict: Результаты идентификации и распознавания эмоции
    """
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
        
        # Временно блокируем модели от перезаписи во время процесса идентификации
        # Не устанавливаем флаг is_training, так как он должен отражать только процесс обучения
        
        # Идентификация пользователя
        identity = predict_user(features_list_id)
        
        # Распознавание эмоции
        detected_emotion = predict_emotion(features_list_emotion)
        
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
