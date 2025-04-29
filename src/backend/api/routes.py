from flask import Blueprint, request, jsonify, Response
import random
from typing import Tuple, Any, Dict, Union, Optional, List
from backend.ml.voice_identification_model import VoiceIdentificationModel
from backend.ml.emotions_recognitions_model import EmotionRecognitionModel
from werkzeug.datastructures import FileStorage
from backend.api.error_logger import error_logger
from backend.config import EMOTIONS
from backend.api.info_logger import info_logger

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

@api_bp.route('/id_training', methods=['POST'])
def id_training() -> Tuple[Response, int]:
    """
    Эндпоинт для обучения модели распознавания по голосу
    """
    if 'audio' not in request.files or 'name' not in request.form:
        error_logger.log_exception(
            ValueError("Аудиофайл и имя обязательны"),
            "api",
            "voice_identification",
            "Проверка входных данных"
        )
        return jsonify({'error': 'Аудиофайл и имя обязательны'}), 400
    
    audio_file: FileStorage = request.files['audio']
    name: str = request.form['name'].strip()  # Удаляем лишние пробелы
    
    if not name:
        error_logger.log_exception(
            ValueError("Имя не может быть пустым"),
            "api",
            "voice_identification",
            "Проверка имени пользователя"
        )
        return jsonify({'error': 'Имя не может быть пустым'}), 400
    
    if audio_file.filename == '':
        error_logger.log_exception(
            ValueError("Файл не выбран"),
            "api",
            "voice_identification",
            "Проверка наличия файла"
        )
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Передаем аудиофайл напрямую в модель
    result: bool = voice_id_model.train([audio_file], [name])
    
    if result:
        return jsonify({'message': 'Обучение модели завершено успешно'}), 200
    else:
        return jsonify({'error': 'Ошибка при обучении модели'}), 500

@api_bp.route('/em_training', methods=['POST'])
def em_training() -> Tuple[Response, int]:
    """
    Эндпоинт для обучения модели распознавания эмоций
    """
    if 'audio' not in request.files or 'emotion' not in request.form:
        error_logger.log_exception(
            ValueError("Аудиофайл и эмоция обязательны"),
            "api",
            "emotion_recognition",
            "Проверка входных данных"
        )
        return jsonify({'error': 'Аудиофайл и эмоция обязательны'}), 400
    
    audio_file: FileStorage = request.files['audio']
    emotion: str = request.form['emotion'].strip()  # Удаляем лишние пробелы
    
    if not emotion:
        error_logger.log_exception(
            ValueError("Эмоция не может быть пустой"),
            "api",
            "emotion_recognition",
            "Проверка эмоции"
        )
        return jsonify({'error': 'Эмоция не может быть пустой'}), 400
    
    if audio_file.filename == '':
        error_logger.log_exception(
            ValueError("Файл не выбран"),
            "api",
            "emotion_recognition",
            "Проверка наличия файла"
        )
        return jsonify({'error': 'Файл не выбран'}), 400
        
    # Передаем аудиофайл напрямую в модель
    result: bool = emotion_model.train([audio_file], [emotion])
    
    if result:
        return jsonify({'message': 'Обучение модели завершено успешно'}), 200
    else:
        return jsonify({'error': 'Ошибка при обучении модели'}), 500

@api_bp.route('/identify', methods=['POST'])
def identify() -> Response:
    """
    Идентификация пользователя и проверка эмоции по аудиофайлу
    """
    info_logger.info("---Start identification process in API---")
    try:
        # Проверка наличия файла в запросе
        info_logger.info("Checking for audio file in request")
        if 'audio' not in request.files:
            info_logger.info("No audio file provided in request")
            return jsonify({
                'success': False,
                'message': 'Аудиофайл не предоставлен',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Получение файла и параметров
        info_logger.info("Getting audio file and parameters")
        audio_file: FileStorage = request.files['audio']
        expected_emotion: Optional[str] = request.form.get('expected_emotion', None)
        
        if not expected_emotion:
            info_logger.info("No expected emotion provided")
            return jsonify({
                'success': False,
                'message': 'Ожидаемая эмоция не указана',
                'identity': None,
                'emotion': None,
                'match': False
            })
            
        # Проверка, что модели обучены
        info_logger.info("Checking if models are trained")
        if not voice_id_model.is_trained:
            info_logger.info("Voice identification model is not trained")
            return jsonify({
                'success': False,
                'message': 'Модель идентификации не обучена',
                'identity': None,
                'emotion': None,
                'match': False
            })
            
        if not emotion_model.is_trained:
            info_logger.info("Emotion recognition model is not trained")
            return jsonify({
                'success': False,
                'message': 'Модель эмоций не обучена',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Напрямую идентифицируем пользователя по голосу
        info_logger.info("Starting voice identification")
        identity: str = voice_id_model.predict(audio_file)
        info_logger.info(f"Voice identification result: {identity}")
        
        # Получаем результат сравнения эмоций
        info_logger.info("Starting emotion comparison")
        emotion_match: bool = emotion_model.compare_emotion(audio_file, expected_emotion)
        info_logger.info(f"Emotion comparison result: {emotion_match}")
        
        # Получаем распознанную эмоцию
        info_logger.info("Getting detected emotion")
        detected_emotion: Union[str, Dict[str, Union[str, float]]] = emotion_model.predict(audio_file)
        
        # Извлекаем название эмоции если результат - словарь
        emotion_name: str = detected_emotion if isinstance(detected_emotion, str) else detected_emotion.get("emotion", "unknown")
        info_logger.info(f"Detected emotion: {emotion_name}")
        
        # Рассчитываем успешность идентификации
        info_logger.info("Calculating identification success")
        success: bool = True
        message: str = "Идентификация выполнена успешно"
        
        if identity == "unknown" and emotion_name == "unknown":
            success = False
            message = "Не удалось распознать пользователя и эмоцию"
            info_logger.info("Failed to recognize both user and emotion")
        elif identity == "unknown":
            success = False
            message = "Не удалось распознать пользователя"
            info_logger.info("Failed to recognize user")
        elif emotion_name == "unknown":
            success = False
            message = "Не удалось распознать эмоцию"
            info_logger.info("Failed to recognize emotion")
        elif not emotion_match:
            success = False
            message = f"Эмоция не соответствует ожидаемой ({emotion_name} вместо {expected_emotion})"
            info_logger.info(f"Emotion mismatch: {emotion_name} instead of {expected_emotion}")
        
        info_logger.info(f"Final identification result - Success: {success}, Message: {message}")
        return jsonify({
            'success': success,
            'message': message,
            'identity': identity,
            'emotion': emotion_name,
            'match': emotion_match
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
    finally:
        info_logger.info("---End identification process in API---")

@api_bp.route('/status', methods=['GET'])
def get_status() -> Tuple[Response, int]:
    """
    Эндпоинт для получения статуса моделей
    """
    try:
        return jsonify({
            'voice_id_training': voice_id_model.is_training,
            'emotion_training': emotion_model.is_training
        }), 200
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "get_status",
            "Ошибка при получении статуса моделей"
        )
        return jsonify({'error': f'Ошибка получения статуса: {str(e)}'}), 500

@api_bp.route('/daily_emotion', methods=['GET'])
def get_daily_emotion_endpoint() -> Tuple[Response, int]:
    """
    Возвращает эмоцию дня, которая остается постоянной до перезапуска сервера
    """
    return jsonify({'emotion': DAILY_EMOTION}), 200

