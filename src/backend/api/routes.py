from flask import Blueprint, request, jsonify
from backend.processors.processors_main import (create_voice_id_training_dataset_from_audio, create_emotion_training_dataset_from_audio, 
                                                get_audio_features_from_audio_for_id_user_prediction, get_audio_features_from_audio_for_emotion_prediction
                                                )
from backend.ml.ml_main import (
    get_daily_emotion, 
    get_training_status, 
    get_training_progress, 
    train_voice_id_model, 
    train_emotion_model, 
    identify_with_emotion
)
from backend.api.error_logger import error_logger

def handle_error(error, module="api", location="general", status_code=400):
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
    error_message = str(error)
    
    # Логирование ошибки
    error_logger.log_error(error_message, module, location)
    
    # Возвращаем JSON-ответ с сообщением об ошибке
    return jsonify({'error': error_message}), status_code

api_bp = Blueprint('api', __name__)

# Выводим эмоцию дня из ml_main для информации
print(f"Эмоция дня установлена: {get_daily_emotion()}")

@api_bp.route('/id_training', methods=['POST'])
def id_training():
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
    
    audio_file = request.files['audio']
    name = request.form['name'].strip()  # Удаляем лишние пробелы
    
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
    
    # Обработка и подготовка данных
    dataset = create_voice_id_training_dataset_from_audio(audio_file, name)
    
    # Запускаем обучение через ml_main
    result = train_voice_id_model(dataset)
    if result:
        return jsonify({'message': 'Обучение модели завершено успешно'}), 200
    else:
        return jsonify({'error': 'Ошибка при обучении модели'}), 500

@api_bp.route('/em_training', methods=['POST'])
def em_training():
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
    
    audio_file = request.files['audio']
    emotion = request.form['emotion'].strip()  # Удаляем лишние пробелы
    
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

    # Обработка и подготовка данных
    dataset = create_emotion_training_dataset_from_audio(audio_file, emotion)
    
    # Запускаем обучение через ml_main
    result = train_emotion_model(dataset)
    if result:
        return jsonify({'message': 'Обучение модели завершено успешно'}), 200
    else:
        return jsonify({'error': 'Ошибка при обучении модели'}), 500

@api_bp.route('/identify', methods=['POST'])
def identify():
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
        audio_file = request.files['audio']
        expected_emotion = request.form.get('expected_emotion', None)
        
        if not expected_emotion:
            return jsonify({
                'success': False,
                'message': 'Ожидаемая эмоция не указана',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Получение признаков для идентификации пользователя и эмоции
        features_list_for_identification = get_audio_features_from_audio_for_id_user_prediction(audio_file)
        features_list_for_emotion = get_audio_features_from_audio_for_emotion_prediction(audio_file)
        
        # Используем функцию identify_with_emotion из ml_main
        result = identify_with_emotion(features_list_for_identification, features_list_for_emotion, expected_emotion)
        
        return jsonify(result)
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "identification",
            "Ошибка при идентификации пользователя"
        )
        return jsonify({
            'success': False,
            'message': f'Внутренняя ошибка: {str(e)}',
            'identity': None,
            'emotion': None,
            'match': False
        })

@api_bp.route('/status', methods=['GET'])
def get_status():
    """
    Эндпоинт для получения статуса моделей
    """
    try:
        return jsonify(get_training_status()), 200
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "get_status",
            "Ошибка при получении статуса моделей"
        )
        return jsonify({'error': f'Ошибка получения статуса: {str(e)}'}), 500

@api_bp.route('/daily_emotion', methods=['GET'])
def get_daily_emotion_endpoint():
    """
    Возвращает эмоцию дня, которая остается постоянной до перезапуска сервера
    """
    return jsonify({'emotion': get_daily_emotion()}), 200

@api_bp.route('/training_progress', methods=['GET'])
def get_training_progress_endpoint():
    """
    Эндпоинт для получения прогресса обучения моделей
    """
    model_type = request.args.get('model_type', 'all')
    
    if model_type not in ['all', 'voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
        
    progress = get_training_progress(model_type)
    if progress is not None:
        return jsonify(progress), 200
    else:
        return jsonify({'error': 'Ошибка получения прогресса'}), 500

