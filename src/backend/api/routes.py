from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import time
import random
from backend.processors.audio_processor import process_audio, enhanced_silence_removal, enhanced_noise_removal, improved_split_audio as split_audio
from backend.processors.dataset_creator import create_voice_id_dataset, create_emotion_dataset
from backend.voice_identification.model import VoiceIdentificationModel
from backend.emotion_recognition.model import EmotionRecognitionModel
from backend.api.error_logger import error_logger
from backend.config import SAMPLE_RATE, EMOTIONS
import zipfile
import io
import sys
import librosa
from datetime import datetime

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
    try:
        processed_audio = process_audio(audio_file)
        dataset = create_voice_id_dataset(processed_audio, name)
        
        # Проверяем, не обучается ли модель в данный момент
        if voice_id_model.is_training:
            return jsonify({'error': 'Модель уже обучается. Попробуйте позже'}), 429
            
        # Запускаем обучение напрямую
        result = train_voice_id_model(dataset)
        if result:
            return jsonify({'message': 'Обучение модели завершено успешно'}), 200
        else:
            return jsonify({'error': 'Ошибка при обучении модели'}), 500
        
    except ValueError as e:
        error_logger.log_exception(
            e,
            "api",
            "voice_identification",
            "Ошибка валидации данных"
        )
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "voice_identification",
            "Неожиданная ошибка при обработке данных"
        )
        return jsonify({'error': f'Ошибка обработки данных: {str(e)}'}), 500

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
    try:
        processed_audio = process_audio(audio_file)
        dataset = create_emotion_dataset(processed_audio, emotion)
        
        # Проверяем, не обучается ли модель в данный момент
        if emotion_model.is_training:
            return jsonify({'error': 'Модель уже обучается. Попробуйте позже'}), 429
            
        # Запускаем обучение напрямую
        result = train_emotion_model(dataset)
        if result:
            return jsonify({'message': 'Обучение модели завершено успешно'}), 200
        else:
            return jsonify({'error': 'Ошибка при обучении модели'}), 500
            
    except ValueError as e:
        error_logger.log_exception(
            e,
            "api",
            "emotion_recognition",
            "Ошибка валидации данных"
        )
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "emotion_recognition",
            "Неожиданная ошибка при обработке данных"
        )
        return jsonify({'error': f'Ошибка обработки данных: {str(e)}'}), 500

@api_bp.route('/identify', methods=['POST'])
def identify():
    """
    Идентификация пользователя и проверка эмоции по аудиофайлу
    """
    # Блокировка доступа к моделям во время идентификации
    voice_id_model.is_training = True
    emotion_model.is_training = True
    
    try:
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
        
        # Чтение аудиофайла
        audio_bytes = audio_file.read()
        
        # Обработка аудио 
        try:
            audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
            # Удаление шума и тишины
            audio_data = enhanced_noise_removal(audio_data)
            audio_data = enhanced_silence_removal(audio_data, SAMPLE_RATE)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Ошибка обработки аудио: {str(e)}',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Дробление аудио на фрагменты
        audio_fragments = split_audio(audio_data)
        
        if not audio_fragments or len(audio_fragments) == 0:
            return jsonify({
                'success': False,
                'message': 'Не удалось выделить значимые фрагменты из аудио',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Идентификация пользователя
        voice_results = voice_id_model.predict(audio_fragments)
        
        # Для идентификации голоса: голосование по всем фрагментам
        if not voice_results:
            return jsonify({
                'success': False,
                'message': 'Не удалось выполнить идентификацию',
                'identity': None,
                'emotion': None,
                'match': False
            })
        
        # Голосование по результатам фрагментов
        vote_counts = {}
        confidence_sums = {}
        
        for result in voice_results:
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
                identity = "unknown"
        else:
            identity = "unknown"
        
        # Распознавание эмоции
        emotion_results = emotion_model.predict(audio_fragments)
        
        # Голосование по результатам фрагментов для эмоций
        emotion_vote_counts = {}
        emotion_confidence_sums = {}
        
        for result in emotion_results:
            label = result['label']
            confidence = result['confidence']
            
            if label not in emotion_vote_counts:
                emotion_vote_counts[label] = 0
                emotion_confidence_sums[label] = 0
            
            emotion_vote_counts[label] += 1
            emotion_confidence_sums[label] += confidence
        
        # Находим эмоцию с наибольшим количеством голосов
        if emotion_vote_counts:
            detected_emotion = max(emotion_vote_counts.keys(), key=lambda k: emotion_vote_counts[k])
            emotion_avg_confidence = emotion_confidence_sums[detected_emotion] / emotion_vote_counts[detected_emotion]
            
            # Если средняя уверенность низкая, считаем неизвестной
            if emotion_avg_confidence < 0.6:
                detected_emotion = "unknown"
        else:
            detected_emotion = "unknown"
        
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
        
        return jsonify({
            'success': success,
            'message': message,
            'identity': identity,
            'emotion': detected_emotion,
            'match': emotion_match
        })
    except Exception as e:
        error_logger.log_exception(
            e,
            "api",
            "identification",
            "processing",
            "Ошибка при идентификации пользователя"
        )
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        error_message = f"{fname} - {line_no} - {str(e)}"
        print(error_message)
        error_logger.log_error(f"Ошибка при идентификации: {error_message}", "api", "identify")
        return jsonify({
            'success': False,
            'message': f'Внутренняя ошибка: {str(e)}',
            'identity': None,
            'emotion': None,
            'match': False
        })
    finally:
        voice_id_model.is_training = False
        emotion_model.is_training = False

@api_bp.route('/model/reset', methods=['POST'])
def reset_model():
    """
    Сброс модели до начального состояния
    """
    model_type = request.json.get('model_type')
    
    if model_type not in ['voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
    
    try:
        if model_type == 'voice_id':
            voice_id_model.reset()
            return jsonify({'message': 'Модель идентификации голоса успешно сброшена'}), 200
        else:
            emotion_model.reset()
            return jsonify({'message': 'Модель распознавания эмоций успешно сброшена'}), 200
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        return jsonify({'error': f'Ошибка сброса модели: {str(e)}'}), 500

@api_bp.route('/model/load', methods=['POST'])
def load_model():
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
    
    # Проверка существования файла
    if not os.path.exists(f'{file_path}.h5') or not os.path.exists(f'{file_path}_metadata.pkl'):
        return jsonify({'error': 'Файл модели не найден'}), 404
    
    try:
        if model_type == 'voice_id':
            voice_id_model.load(file_path)
            return jsonify({'message': 'Модель идентификации голоса успешно загружена'}), 200
        else:
            emotion_model.load(file_path)
            return jsonify({'message': 'Модель распознавания эмоций успешно загружена'}), 200
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
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
            save_dir = os.path.join(base_dir, 'models', 'voice_identification')
        else:
            save_dir = os.path.join(base_dir, 'models', 'emotion_recognition')
        
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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        return jsonify({'error': f'Ошибка загрузки файла: {str(e)}'}), 500

@api_bp.route('/status', methods=['GET'])
def get_status():
    """
    Эндпоинт для получения статуса моделей
    """
    voice_id_status = True
    emotion_status = True
    
    try:
        voice_id_status = voice_id_model.is_training
        emotion_status = emotion_model.is_training
    except:
        pass
    
    return jsonify({
        'voice_id_training': not voice_id_status,
        'emotion_training': not emotion_status
    }), 200

@api_bp.route('/daily_emotion', methods=['GET'])
def get_daily_emotion():
    """
    Возвращает эмоцию дня, которая остается постоянной до перезапуска сервера
    """
    return jsonify({'emotion': DAILY_EMOTION}), 200

@api_bp.route('/training_progress', methods=['GET'])
def get_training_progress():
    """
    Эндпоинт для получения прогресса обучения моделей
    """
    model_type = request.args.get('model_type', 'all')
    
    if model_type == 'all':
        return jsonify(training_progress), 200
    elif model_type in ['voice_id', 'emotion']:
        return jsonify(training_progress[model_type]), 200
    else:
        return jsonify({'error': 'Неверный тип модели'}), 400

@api_bp.route('/model/download', methods=['POST'])
def download_model():
    """
    Сохранение модели и отправка файлов пользователю для скачивания
    """
    model_type = request.json.get('model_type')
    
    if model_type not in ['voice_id', 'emotion']:
        return jsonify({'error': 'Неверный тип модели'}), 400
    
    try:
        memory_file = io.BytesIO()
        
        if model_type == 'voice_id':
            if voice_id_model.is_training:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
            
            # Сохраняем модель
            file_path = voice_id_model.save()
            if not file_path:
                return jsonify({'error': 'Модель не обучена или не может быть сохранена'}), 400
            
            # Создаем zip-архив в памяти
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                h5_file = f"{file_path}.h5"
                metadata_file = f"{file_path}_metadata.pkl"
                
                # Добавляем файлы в архив
                zf.write(h5_file, os.path.basename(h5_file))
                zf.write(metadata_file, os.path.basename(metadata_file))
            
            # Перемещаем указатель в начало файла для чтения
            memory_file.seek(0)
            
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'voice_id_model_{int(time.time())}.zip'
            )
        else:  # emotion
            if emotion_model.is_training:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
            
            # Сохраняем модель
            file_path = emotion_model.save()
            if not file_path:
                return jsonify({'error': 'Модель не обучена или не может быть сохранена'}), 400
            
            # Создаем zip-архив в памяти
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                h5_file = f"{file_path}.h5"
                metadata_file = f"{file_path}_metadata.pkl"
                
                # Добавляем файлы в архив
                zf.write(h5_file, os.path.basename(h5_file))
                zf.write(metadata_file, os.path.basename(metadata_file))
            
            # Перемещаем указатель в начало файла для чтения
            memory_file.seek(0)
            
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'emotion_model_{int(time.time())}.zip'
            )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        return jsonify({'error': f'Ошибка при скачивании модели: {str(e)}'}), 500

def train_voice_id_model(dataset):
    """
    Обучает модель идентификации голоса.
    
    Args:
        dataset: Датасет для обучения модели.
        
    Returns:
        bool: успешность обучения модели
    """
    try:
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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        # Обновляем статус обучения при ошибке
        training_progress['emotion']['status'] = 'error'
        return False
