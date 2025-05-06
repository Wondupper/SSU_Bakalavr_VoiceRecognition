from flask import Blueprint, request, jsonify, Response
import random
from typing import Tuple, Any, Dict, Union, Optional, List
from backend.ml.voice_identification.model import VoiceIdentificationModel
from backend.ml.emotions_recognition.model import EmotionRecognitionModel
from werkzeug.datastructures import FileStorage
from src.backend.loggers.error_logger import error_logger
from src.backend.config import EMOTIONS
from src.backend.loggers.info_logger import info_logger

# Инициализация моделей
voice_id_model: VoiceIdentificationModel = VoiceIdentificationModel()
emotion_model: EmotionRecognitionModel = EmotionRecognitionModel()

# Эмоция дня, которая генерируется один раз при запуске сервера
DAILY_EMOTION: str = random.choice(EMOTIONS)

def handle_error(error: Any, module: str = "api", location: str = "general", status_code: int = 400) -> Tuple[Response, int]:
    """
    Обработчик ошибок, который логирует ошибку и возвращает соответствующий JSON-ответ
    
    Args:
        error: Объект ошибки или строка с описанием ошибки
        module: Название модуля, где произошла ошибка
        location: Конкретное место в коде, где произошла ошибка
        status_code: HTTP-код ответа (по умолчанию 400)
        
    Returns:
        Tuple: (JSON-ответ, статус-код)
    """
    error_message: str = str(error)
    
    # Логирование ошибки
    error_logger.log_error(error_message, module, location)
    
    # Возвращаем JSON-ответ с сообщением об ошибке
    return jsonify({'error': error_message}), status_code

api_bp: Blueprint = Blueprint('api', __name__)

# Выводим эмоцию дня для информации
print(f"Эмоция дня установлена: {DAILY_EMOTION}")


@api_bp.route('/identify', methods=['POST'])
def identify() -> Response:
    """
    Идентификация пользователя и проверка эмоции по аудиофайлу
    """
    try:
        # Проверка наличия файла в запросе
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Аудиофайл не предоставлен',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Получение файла и параметров
        audio_file: FileStorage = request.files['audio']
        expected_emotion: Optional[str] = request.form.get('expected_emotion', None)
        
        if not expected_emotion:
            return jsonify({
                'success': False,
                'message': 'Ожидаемая эмоция не указана',
                'identity': None,
                'emotion': None,
                'match': False
            })
            
        # Проверка, что модели обучены
        if not voice_id_model.is_trained:
            return jsonify({
                'success': False,
                'message': 'Модель идентификации не обучена',
                'identity': None,
                'emotion': None,
                'match': False
            })
            
        if not emotion_model.is_trained:
            return jsonify({
                'success': False,
                'message': 'Модель эмоций не обучена',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Напрямую идентифицируем пользователя по голосу
        identity: str = voice_id_model.predict(audio_file)
        
        
        # Получаем распознанную эмоцию
        detected_emotion: str = emotion_model.predict(audio_file)
        
        # Рассчитываем успешность идентификации
        success: bool = True
        message: str = "Идентификация выполнена успешно"
        
        if identity == "unknown" and detected_emotion != expected_emotion:
            success = False
            message = "Не удалось распознать пользователя и эмоцию"
            info_logger.info("Failed to recognize both user and emotion")
        elif identity == "unknown":
            success = False
            message = "Не удалось распознать пользователя"
            info_logger.info("Failed to recognize user")
        elif expected_emotion != detected_emotion:
            success = False
            message = f"Эмоция не соответствует ожидаемой ({detected_emotion} вместо {expected_emotion})"
            info_logger.info(f"Emotion mismatch: {detected_emotion} instead of {expected_emotion}")
        
        info_logger.info(f"Final identification result - Success: {success}, Message: {message}")
        return jsonify({
            'success': success,
            'message': message,
            'identity': identity,
            'emotion': expected_emotion,
            'match': detected_emotion
        })
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "identification",
            "Ошибка при идентификации пользователя"
        )
        info_logger.info(f"Error during identification: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Внутренняя ошибка: {str(e)}',
            'identity': None,
            'emotion': None,
            'match': False
        })


@api_bp.route('/daily_emotion', methods=['GET'])
def get_daily_emotion_endpoint() -> Tuple[Response, int]:
    """
    Возвращает эмоцию дня, которая остается постоянной до перезапуска сервера
    """
    return jsonify({'emotion': DAILY_EMOTION}), 200

