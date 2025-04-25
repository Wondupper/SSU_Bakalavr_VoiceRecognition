from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import time
import random
from backend.processors.processors_main import create_voice_id_dataset_from_audio, create_emotion_dataset_from_audio, get_audio_fragments_from_audio
from backend.ml.ml_main import (
    get_daily_emotion, 
    get_training_status, 
    get_training_progress, 
    train_voice_id_model, 
    train_emotion_model, 
    identify_with_emotion, 
    reset_voice_id_model,
    reset_emotion_model,
    load_voice_id_model,
    load_emotion_model,
    save_voice_id_model,
    save_emotion_model
)
from backend.api.error_logger import error_logger
from backend.config import EMOTIONS, VOICE_ID_MODELS_DIR, EMOTION_MODELS_DIR
import zipfile
import io

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

# Эмоция дня, которая генерируется один раз при запуске сервера
DAILY_EMOTION = random.choice(EMOTIONS)
print(f"Эмоция дня установлена: {DAILY_EMOTION}")

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
    dataset = create_voice_id_dataset_from_audio(audio_file, name)
    
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
    dataset = create_emotion_dataset_from_audio(audio_file, emotion)
    
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
        
        # Дробление аудио на фрагменты
        audio_fragments = get_audio_fragments_from_audio(audio_file)
        
        # Используем функцию identify_with_emotion из ml_main
        result = identify_with_emotion(audio_fragments, expected_emotion)
        
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

@api_bp.route('/model/reset', methods=['POST'])
def reset_model_endpoint():
    """
    Сброс модели до начального состояния
    """
    model_type = request.json.get('model_type')
    
    if model_type not in ['voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
    
    try:
        if model_type == 'voice_id':
            result = reset_voice_id_model()
        else:  # model_type == 'emotion'
            result = reset_emotion_model()
            
        if result:
            return jsonify({'message': f'Модель {model_type} успешно сброшена'}), 200
        else:
            return jsonify({'error': 'Ошибка сброса модели'}), 500
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "reset_model",
            f"Ошибка при сбросе модели {model_type}"
        )
        return jsonify({'error': f'Ошибка сброса модели: {str(e)}'}), 500

@api_bp.route('/model/load', methods=['POST'])
def load_model_endpoint():
    """
    Загрузка модели из файла
    """
    model_type = request.json.get('model_type')
    file_path = request.json.get('file_path')
    
    if model_type not in ['voice_id', 'emotion'] or not file_path:
        return jsonify({'error': 'Неверный тип модели или путь'}), 400
    
    # Добавляем дополнительную проверку пути для безопасности
    # Убедимся, что путь находится в директории models и не содержит "../"
    if '..' in file_path or not file_path.startswith(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')):
        return jsonify({'error': 'Недопустимый путь к файлу'}), 400
    
    try:
        if model_type == 'voice_id':
            result = load_voice_id_model(file_path)
        else:  # model_type == 'emotion'
            result = load_emotion_model(file_path)
            
        if result:
            return jsonify({'message': f'Модель {model_type} успешно загружена'}), 200
        else:
            return jsonify({'error': 'Ошибка загрузки модели'}), 500
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "load_model",
            f"Ошибка при загрузке модели {model_type}"
        )
        return jsonify({'error': f'Ошибка загрузки модели: {str(e)}'}), 500

@api_bp.route('/model/upload', methods=['POST'])
def upload_model():
    """
    Загрузка файла модели на сервер
    """
    if 'model_file' not in request.files:
        return jsonify({'error': 'Файл модели не найден'}), 400
    
    model_file = request.files['model_file']
    
    # Получаем тип модели из формы, а не из JSON
    model_type = request.form.get('model_type')
    
    if model_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if model_type not in ['voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
    
    # Проверка расширения файла
    if not model_file.filename.endswith('.h5') and not model_file.filename.endswith('.pkl'):
        return jsonify({'error': 'Неверный формат файла. Допустимые: .h5, .pkl'}), 400
    
    try:
        # Определяем директорию сохранения в зависимости от типа модели
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if model_type == 'voice_id':
            save_dir = os.path.join(base_dir, VOICE_ID_MODELS_DIR)
        else:
            save_dir = os.path.join(base_dir, EMOTION_MODELS_DIR)
        
        # Проверяем/создаем директорию
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Создаем безопасное имя файла
        filename = secure_filename(model_file.filename)
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        
        # Полный путь к файлу
        file_path = os.path.join(save_dir, safe_filename)
        
        # Сохраняем файл
        model_file.save(file_path)
        
        # Проверяем, что файл действительно сохранен и доступен
        if not os.path.exists(file_path):
            return jsonify({'error': 'Ошибка сохранения файла на сервере'}), 500
            
        # Возвращаем путь без расширения для последующей загрузки
        base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
        
        return jsonify({
            'message': 'Файл успешно загружен',
            'file_path': base_path
        }), 200
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "upload_model",
            f"Ошибка загрузки файла: {str(e)}"
        )
        return jsonify({'error': f'Ошибка загрузки файла: {str(e)}'}), 500

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

@api_bp.route('/model/download', methods=['POST'])
def download_model():
    """
    Сохранение модели и отправка файлов пользователю для скачивания
    """
    model_type = request.json.get('model_type')
    
    if model_type not in ['voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
    
    try:
        # Сохраняем модель с помощью функции из ml_main
        if model_type == 'voice_id':
            file_path = save_voice_id_model()
        else:  # model_type == 'emotion'
            file_path = save_emotion_model()
        
        if not file_path:
            return jsonify({'error': 'Модель не обучена или не может быть сохранена'}), 400
            
        memory_file = io.BytesIO()
        
        # Создаем zip-архив в памяти
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            h5_file = f"{file_path}.h5"
            metadata_file = f"{file_path}_metadata.pkl"
            
            # Добавляем файлы в архив
            zf.write(h5_file, os.path.basename(h5_file))
            zf.write(metadata_file, os.path.basename(metadata_file))
        
        # Перемещаем указатель в начало файла для чтения
        memory_file.seek(0)
        
        # Формируем имя для скачиваемого файла
        download_name = f'{model_type}_model_{int(time.time())}.zip'
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=download_name
        )
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "download_model",
            f"Ошибка при скачивании модели {model_type}"
        )
        return jsonify({'error': f'Ошибка при скачивании модели: {str(e)}'}), 500
