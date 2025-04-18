from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import threading
import time
import random
from backend.processors.audio_processor import process_audio
from backend.processors.dataset_creator import create_voice_id_dataset, create_emotion_dataset
from backend.voice_identification.model import VoiceIdentificationModel
from backend.emotion_recognition.model import EmotionRecognitionModel
from backend.api.error_logger import error_logger
import zipfile
import io
import sys

api_bp = Blueprint('api', __name__)

# Инициализация моделей
voice_id_model = VoiceIdentificationModel()
emotion_model = EmotionRecognitionModel()

# Блокировка для обучения моделей
voice_id_model_lock = threading.Lock()
emotion_model_lock = threading.Lock()

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
EMOTIONS = ['гнев', 'радость', 'грусть']
DAILY_EMOTION = random.choice(EMOTIONS)
print(f"Эмоция дня установлена: {DAILY_EMOTION}")

@api_bp.route('/id_training', methods=['POST'])
def id_training():
    """
    Эндпоинт для обучения модели распознавания по голосу
    """
    if 'audio' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Аудиофайл и имя обязательны'}), 400
    
    audio_file = request.files['audio']
    name = request.form['name'].strip()  # Удаляем лишние пробелы
    
    if not name:
        return jsonify({'error': 'Имя не может быть пустым'}), 400
    
    if audio_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Ограничение максимального размера файла (например, 20MB)
    if request.content_length > 20 * 1024 * 1024:
        return jsonify({'error': 'Размер файла превышает допустимый предел (20MB)'}), 413
    
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
            except Exception as e:
                voice_id_model_lock.release()
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                error_logger.log_error(f"Не удалось запустить поток обучения: {str(e)}", "training", "voice_id")
                return jsonify({'error': f'Ошибка запуска обучения: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Модель уже обучается. Попробуйте позже'}), 429
    except ValueError as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        error_logger.log_error(f"Неожиданная ошибка при обучении: {str(e)}", "training", "voice_id")
        return jsonify({'error': f'Ошибка обработки данных: {str(e)}'}), 500

@api_bp.route('/em_training', methods=['POST'])
def em_training():
    """
    Эндпоинт для обучения модели распознавания эмоций
    """
    if 'audio' not in request.files or 'emotion' not in request.form:
        return jsonify({'error': 'Аудиофайл и эмоция обязательны'}), 400
    
    audio_file = request.files['audio']
    emotion = request.form['emotion'].strip()  # Удаляем лишние пробелы
    
    if not emotion:
        return jsonify({'error': 'Эмоция не может быть пустой'}), 400
    
    if audio_file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Ограничение максимального размера файла (например, 20MB)
    if request.content_length > 20 * 1024 * 1024:
        return jsonify({'error': 'Размер файла превышает допустимый предел (20MB)'}), 413
    
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
            except Exception as e:
                emotion_model_lock.release()
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                error_logger.log_error(f"Не удалось запустить поток обучения эмоций: {str(e)}", "training", "emotion")
                return jsonify({'error': f'Ошибка запуска обучения: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Модель уже обучается. Попробуйте позже'}), 429
    except ValueError as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        error_logger.log_error(f"Неожиданная ошибка при обучении модели эмоций: {str(e)}", "training", "emotion")
        return jsonify({'error': f'Ошибка обработки данных: {str(e)}'}), 500

@api_bp.route('/identify', methods=['POST'])
def identify():
    """
    Эндпоинт для идентификации пользователя по голосу
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
        # Обработка аудио
        processed_audio = process_audio(audio_file)
        
        # Пытаемся идентифицировать пользователя
        try:
            user_name = voice_id_model.predict(processed_audio)
            # "unknown" здесь означает легитимный результат - голос не распознан
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            error_logger.log_error(f"Ошибка при идентификации пользователя: {str(e)}", "api", "identify")
            # Только при технической ошибке в модели возвращаем unknown
            user_name = "unknown"
        
        # Пытаемся определить эмоцию
        try:
            detected_emotion = emotion_model.predict(processed_audio)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            error_logger.log_error(f"Ошибка при определении эмоции: {str(e)}", "api", "identify")
            # При технической ошибке в модели выбираем наиболее нейтральную эмоцию
            detected_emotion = "радость" # Выбираем эмоцию по умолчанию
        
        # Проверка совпадения эмоции
        emotion_match = detected_emotion == expected_emotion
        
        return jsonify({
            'user': user_name,
            'emotion_match': emotion_match,
            'detected_emotion': detected_emotion  # Добавляем обнаруженную эмоцию в ответ
        }), 200
        
    except ValueError as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        # Обработка ошибок валидации
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        # Обработка других ошибок
        error_logger.log_error(f"Общая ошибка при идентификации: {str(e)}", "api", "identify")
        return jsonify({'error': f'Ошибка обработки запроса: {str(e)}'}), 500

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
            if voice_id_model_lock.acquire(blocking=False):
                try:
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
                finally:
                    voice_id_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
        else:  # emotion
            if emotion_model_lock.acquire(blocking=False):
                try:
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
                finally:
                    emotion_model_lock.release()
            else:
                return jsonify({'error': 'Модель используется. Попробуйте позже'}), 429
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        return jsonify({'error': f'Ошибка при скачивании модели: {str(e)}'}), 500

def train_voice_id_model(dataset):
    """
    Функция для обучения модели идентификации голоса в отдельном потоке с отслеживанием прогресса
    """
    try:
        # Устанавливаем начальные значения для отслеживания прогресса
        training_progress['voice_id']['status'] = 'training'
        training_progress['voice_id']['current_epoch'] = 0
        training_progress['voice_id']['total_epochs'] = 100  # Максимальное число эпох
        training_progress['voice_id']['accuracy'] = 0.0
        training_progress['voice_id']['loss'] = 0.0
        training_progress['voice_id']['start_time'] = time.time()
        
        # Принудительная сборка мусора перед обучением
        import gc
        gc.collect()
        
        # Логируем размер датасета перед обучением
        error_logger.log_error(
            f"Начато обучение модели идентификации. Размер датасета: {len(dataset)} примеров",
            "training", "voice_id"
        )
        
        # Ограничиваем размер датасета для обучения если он слишком большой 
        # (для предотвращения исчерпания памяти)
        MAX_DATASET_SIZE = 500  # Увеличено для полноценного обучения, но с ограничением
        if len(dataset) > MAX_DATASET_SIZE:
            # Берем случайную выборку из датасета
            import random
            dataset = random.sample(dataset, MAX_DATASET_SIZE)
            error_logger.log_error(
                f"Датасет обрезан до {MAX_DATASET_SIZE} элементов для экономии памяти",
                "training", "voice_id"
            )
        
        # Создаем колбэк для отслеживания прогресса
        progress_callback = TrainingProgressCallback('voice_id', training_progress)
        
        # Обучаем модель с колбэком прогресса
        voice_id_model.train(dataset, progress_callback=progress_callback)
        
        # Устанавливаем статус завершения
        training_progress['voice_id']['status'] = 'completed'
        
        # Принудительная сборка мусора после обучения
        gc.collect()
        
        error_logger.log_error(
            "Обучение модели идентификации успешно завершено",
            "training", "voice_id"
        )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        error_logger.log_error(f"Ошибка при обучении модели идентификации: {str(e)}", "training", "voice_id")
        training_progress['voice_id']['status'] = 'error'
    finally:
        voice_id_model_lock.release()  # Всегда освобождаем блокировку

def train_emotion_model(dataset):
    """
    Функция для обучения модели распознавания эмоций в отдельном потоке с отслеживанием прогресса
    """
    try:
        # Устанавливаем начальные значения для отслеживания прогресса
        training_progress['emotion']['status'] = 'training'
        training_progress['emotion']['current_epoch'] = 0
        training_progress['emotion']['total_epochs'] = 100  # Максимальное число эпох
        training_progress['emotion']['accuracy'] = 0.0
        training_progress['emotion']['loss'] = 0.0
        training_progress['emotion']['start_time'] = time.time()
        
        # Принудительная сборка мусора перед обучением
        import gc
        gc.collect()
        
        # Логируем размер датасета перед обучением
        error_logger.log_error(
            f"Начато обучение модели эмоций. Размер датасета: {len(dataset)} примеров",
            "training", "emotion"
        )
        
        # Ограничиваем размер датасета для обучения если он слишком большой 
        # (для предотвращения исчерпания памяти)
        MAX_DATASET_SIZE = 500  # Увеличено для полноценного обучения, но с ограничением
        if len(dataset) > MAX_DATASET_SIZE:
            # Берем случайную выборку из датасета
            import random
            dataset = random.sample(dataset, MAX_DATASET_SIZE)
            error_logger.log_error(
                f"Датасет обрезан до {MAX_DATASET_SIZE} элементов для экономии памяти",
                "training", "emotion"
            )
        
        # Создаем колбэк для отслеживания прогресса
        progress_callback = TrainingProgressCallback('emotion', training_progress)
        
        # Обучаем модель с колбэком прогресса
        emotion_model.train(dataset, progress_callback=progress_callback)
        
        # Устанавливаем статус завершения
        training_progress['emotion']['status'] = 'completed'
        
        # Принудительная сборка мусора после обучения
        gc.collect()
        
        error_logger.log_error(
            "Обучение модели эмоций успешно завершено",
            "training", "emotion"
        )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        error_logger.log_error(f"Ошибка при обучении модели эмоций: {str(e)}", "training", "emotion")
        training_progress['emotion']['status'] = 'error'
    finally:
        emotion_model_lock.release()  # Всегда освобождаем блокировку

# Класс для колбэка отслеживания прогресса
class TrainingProgressCallback:
    def __init__(self, model_type, progress_dict):
        self.model_type = model_type
        self.progress_dict = progress_dict
    
    def on_epoch_end(self, epoch, logs=None):
        """Вызывается после каждой эпохи обучения"""
        logs = logs or {}
        self.progress_dict[self.model_type]['current_epoch'] = epoch + 1
        self.progress_dict[self.model_type]['accuracy'] = float(logs.get('accuracy', 0.0))
        self.progress_dict[self.model_type]['loss'] = float(logs.get('loss', 0.0))