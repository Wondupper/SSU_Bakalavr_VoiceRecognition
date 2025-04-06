from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
import threading
import time
from backend.processors.audio_processor import process_audio
from backend.processors.dataset_creator import create_voice_id_dataset, create_emotion_dataset
from backend.voice_identification.model import VoiceIdentificationModel
from backend.emotion_recognition.model import EmotionRecognitionModel
from backend.api.error_logger import error_logger

api_bp = Blueprint('api', __name__)

# Инициализация моделей
voice_id_model = VoiceIdentificationModel()
emotion_model = EmotionRecognitionModel()

# Блокировка для обучения моделей
voice_id_model_lock = threading.Lock()
emotion_model_lock = threading.Lock()

@api_bp.route('/id_training', methods=['POST'])
def id_training():
    """
    Эндпоинт для обучения модели распознавания по голосу
    """
    if 'audio' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Аудиофайл и имя обязательны'}), 400
    
    audio_file = request.files['audio']
    name = request.form['name']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Обработка и подготовка данных
    try:
        processed_audio = process_audio(audio_file)
        dataset = create_voice_id_dataset(processed_audio, name)
        
        # Проверка блокировки модели и запуск обучения
        if voice_id_model_lock.acquire(blocking=False):
            try:
                # Запустить обучение в отдельном потоке
                threading.Thread(target=train_voice_id_model, args=(dataset,)).start()
                return jsonify({'message': 'Обучение модели начато успешно'}), 200
            finally:
                pass  # Блокировка будет освобождена после обучения
        else:
            return jsonify({'error': 'Модель уже обучается. Попробуйте позже'}), 429
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки данных: {str(e)}'}), 500

@api_bp.route('/em_training', methods=['POST'])
def em_training():
    """
    Эндпоинт для обучения модели распознавания эмоций
    """
    if 'audio' not in request.files or 'emotion' not in request.form:
        return jsonify({'error': 'Аудиофайл и эмоция обязательны'}), 400
    
    audio_file = request.files['audio']
    emotion = request.form['emotion']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if emotion not in ['гнев', 'радость', 'грусть']:
        return jsonify({'error': 'Неверная эмоция. Допустимые: гнев, радость, грусть'}), 400
    
    # Обработка и подготовка данных
    try:
        processed_audio = process_audio(audio_file)
        dataset = create_emotion_dataset(processed_audio, emotion)
        
        # Проверка блокировки модели и запуск обучения
        if emotion_model_lock.acquire(blocking=False):
            try:
                # Запустить обучение в отдельном потоке
                threading.Thread(target=train_emotion_model, args=(dataset,)).start()
                return jsonify({'message': 'Обучение модели начато успешно'}), 200
            finally:
                pass  # Блокировка будет освобождена после обучения
        else:
            return jsonify({'error': 'Модель уже обучается. Попробуйте позже'}), 429
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки данных: {str(e)}'}), 500

@api_bp.route('/identify', methods=['POST'])
def identify():
    """
    Эндпоинт для идентификации пользователя и распознавания эмоций
    """
    if 'audio' not in request.files or 'expected_emotion' not in request.form:
        return jsonify({'error': 'Аудиофайл и ожидаемая эмоция обязательны'}), 400
    
    audio_file = request.files['audio']
    expected_emotion = request.form['expected_emotion']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if expected_emotion not in ['гнев', 'радость', 'грусть']:
        return jsonify({'error': 'Неверная эмоция. Допустимые: гнев, радость, грусть'}), 400
    
    try:
        processed_audio = process_audio(audio_file)
        
        # Идентификация пользователя
        user_name = voice_id_model.predict(processed_audio)
        
        # Распознавание эмоции
        detected_emotion = emotion_model.predict(processed_audio)
        emotion_match = detected_emotion == expected_emotion
        
        return jsonify({
            'user_name': user_name,
            'emotion_match': emotion_match,
            'detected_emotion': detected_emotion
        }), 200
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки данных: {str(e)}'}), 500

@api_bp.route('/panel/reset_model', methods=['POST'])
def reset_model():
    """
    Сброс модели до начального состояния
    """
    model_type = request.json.get('model_type')
    
    if model_type not in ['voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
    
    try:
        if model_type == 'voice_id':
            if voice_id_model_lock.acquire(blocking=False):
                try:
                    voice_id_model.reset()
                    return jsonify({'message': 'Модель идентификации голоса успешно сброшена'}), 200
                finally:
                    voice_id_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
        else:
            if emotion_model_lock.acquire(blocking=False):
                try:
                    emotion_model.reset()
                    return jsonify({'message': 'Модель распознавания эмоций успешно сброшена'}), 200
                finally:
                    emotion_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
    except Exception as e:
        return jsonify({'error': f'Ошибка сброса модели: {str(e)}'}), 500

@api_bp.route('/panel/save_model', methods=['POST'])
def save_model():
    """
    Сохранение модели в файл
    """
    model_type = request.json.get('model_type')
    
    if model_type not in ['voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
    
    try:
        if model_type == 'voice_id':
            if voice_id_model_lock.acquire(blocking=False):
                try:
                    file_path = voice_id_model.save()
                    return jsonify({'message': f'Модель идентификации голоса успешно сохранена в {file_path}'}), 200
                finally:
                    voice_id_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
        else:
            if emotion_model_lock.acquire(blocking=False):
                try:
                    file_path = emotion_model.save()
                    return jsonify({'message': f'Модель распознавания эмоций успешно сохранена в {file_path}'}), 200
                finally:
                    emotion_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
    except Exception as e:
        return jsonify({'error': f'Ошибка сохранения модели: {str(e)}'}), 500

@api_bp.route('/panel/load_model', methods=['POST'])
def load_model():
    """
    Загрузка модели из файла
    """
    model_type = request.json.get('model_type')
    file_path = request.json.get('file_path')
    
    if model_type not in ['voice_id', 'emotion'] or not file_path:
        return jsonify({'error': 'Неверный тип модели или путь'}), 400
    
    # Проверка существования файла
    if not os.path.exists(f'{file_path}.h5') or not os.path.exists(f'{file_path}_metadata.pkl'):
        return jsonify({'error': 'Файл модели не найден'}), 404
    
    try:
        if model_type == 'voice_id':
            if voice_id_model_lock.acquire(blocking=False):
                try:
                    voice_id_model.load(file_path)
                    return jsonify({'message': 'Модель идентификации голоса успешно загружена'}), 200
                finally:
                    voice_id_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
        else:
            if emotion_model_lock.acquire(blocking=False):
                try:
                    emotion_model.load(file_path)
                    return jsonify({'message': 'Модель распознавания эмоций успешно загружена'}), 200
                finally:
                    emotion_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
    except Exception as e:
        return jsonify({'error': f'Ошибка загрузки модели: {str(e)}'}), 500

@api_bp.route('/status', methods=['GET'])
def get_status():
    """
    Эндпоинт для проверки статуса моделей
    """
    # Проверяем статус обучения, используя безопасный способ
    # Если блокировка не может быть получена, значит модель в процессе обучения
    voice_id_status = False
    emotion_status = False
    
    try:
        voice_id_status = voice_id_model_lock.acquire(blocking=False)
    except:
        pass
    finally:
        if voice_id_status:
            voice_id_model_lock.release()
    
    try:
        emotion_status = emotion_model_lock.acquire(blocking=False)
    except:
        pass
    finally:
        if emotion_status:
            emotion_model_lock.release()
    
    return jsonify({
        'voice_id_training': not voice_id_status,
        'emotion_training': not emotion_status
    }), 200

@api_bp.route('/panel/upload_model', methods=['POST'])
def upload_model():
    """
    Загрузка файла модели на сервер
    """
    if 'model_file' not in request.files:
        return jsonify({'error': 'Файл модели не найден'}), 400
    
    model_file = request.files['model_file']
    
    if model_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Проверка расширения файла
    if not model_file.filename.endswith('.h5') and not model_file.filename.endswith('.pkl'):
        return jsonify({'error': 'Неверный формат файла. Допустимые: .h5, .pkl'}), 400
    
    try:
        # Создаем временную директорию для загрузки, если она не существует
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
        
        # Создаем безопасное имя файла
        filename = secure_filename(model_file.filename)
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        
        # Полный путь к файлу
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Сохраняем файл
        model_file.save(file_path)
        
        # Возвращаем путь без расширения для последующей загрузки
        base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
        
        return jsonify({
            'message': 'Файл успешно загружен',
            'file_path': base_path
        }), 200
    except Exception as e:
        return jsonify({'error': f'Ошибка загрузки файла: {str(e)}'}), 500

@api_bp.route('/errors', methods=['GET'])
def get_errors():
    """
    Получение последних ошибок системы
    """
    # Получаем параметр limit из query string, по умолчанию 10
    limit = request.args.get('limit', 10, type=int)
    
    # Получаем последние ошибки
    errors = error_logger.get_recent_errors(limit)
    
    return jsonify({
        'errors': errors
    }), 200

@api_bp.route('/errors/clear', methods=['POST'])
def clear_errors():
    """
    Очистка списка ошибок
    """
    error_logger.clear_errors()
    return jsonify({'message': 'Список ошибок очищен'}), 200

@api_bp.route('/errors/log', methods=['POST'])
def log_error():
    """
    Логирование ошибок с фронтенда
    """
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Неверный формат данных'}), 400
    
    message = data.get('message')
    module = data.get('module', 'frontend')
    location = data.get('location', 'unknown')
    
    error_logger.log_error(
        f"Фронтенд ошибка ({location}): {message}", 
        "frontend", 
        module
    )
    
    return jsonify({'success': True}), 200

def train_voice_id_model(dataset):
    """
    Функция для обучения модели идентификации в отдельном потоке
    """
    try:
        voice_id_model.train(dataset)
    finally:
        voice_id_model_lock.release()

def train_emotion_model(dataset):
    """
    Функция для обучения модели распознавания эмоций в отдельном потоке
    """
    try:
        emotion_model.train(dataset)
    finally:
        emotion_model_lock.release()