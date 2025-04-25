from flask import Flask, send_from_directory, redirect
from backend.api.routes import api_bp, voice_id_model, emotion_model  # Импортируем модели
from backend.ml.ml_main import load_models, save_models
from backend.config import MODELS_DIR, VOICE_ID_MODELS_DIR, EMOTION_MODELS_DIR
from backend.api.error_logger import error_logger
import os
import signal
import atexit
import logging

# Абсолютный путь к директории src
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, static_folder=None)  # Убираем стандартную папку static
app.register_blueprint(api_bp, url_prefix='/api')

# Функция для сохранения моделей при завершении работы
def save_models_on_exit():
    """
    Сохранение моделей перед завершением работы
    """
    print("Сохранение моделей перед завершением работы...")
    save_models(basedir)

# Регистрируем функцию, которая будет вызвана при завершении работы
atexit.register(save_models_on_exit)

# Обработчик сигнала Ctrl+C (SIGINT)
def signal_handler(sig, frame):
    print(f"Получен сигнал {sig}, выполняется корректное завершение работы...")
    save_models_on_exit()
    exit(0)

# Регистрируем обработчики сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Маршруты для фронтенда
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # Главная страница
    if path == "":
        return send_from_directory(os.path.join(basedir, 'frontend/home'), 'index.html')
    
    # Перенаправление устаревшего пути /common.js на новый
    if path == "common.js":
        return redirect("/common/common.js")
    
    # CSS и JavaScript файлы для разных разделов
    if path.endswith('.css') or path.endswith('.js'):
        directory, filename = os.path.split(path)
        return send_from_directory(os.path.join(basedir, 'frontend', directory), filename)
    
    # Специальные разделы сайта
    if path in ['identification', 'idtraining', 'emtraining']:
        return send_from_directory(os.path.join(basedir, f'frontend/{path}'), 'index.html')
    
    # Для всех остальных путей возвращаем главную страницу
    return send_from_directory(os.path.join(basedir, 'frontend/home'), 'index.html')

@app.route('/common/<path:filename>')
def serve_common_files(filename):
    return send_from_directory(os.path.join(basedir, 'frontend/common'), filename)

if __name__ == '__main__':
    # Настройка логирования для Flask
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Подготовка директорий
    print("Подготовка директорий...")
    models_dir = os.path.join(basedir, MODELS_DIR)
    voice_models_dir = os.path.join(basedir, VOICE_ID_MODELS_DIR)
    emotion_models_dir = os.path.join(basedir, EMOTION_MODELS_DIR)

    for directory in [models_dir, voice_models_dir, emotion_models_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Создана директория: {directory}")
            except Exception as e:
                error_logger.log_exception(
                    e,
                    "main",
                    "directory_preparation",
                    f"Ошибка при создании директории {directory}"
                )
                
    # Проверяем права доступа к директориям
    for directory in [voice_models_dir, emotion_models_dir]:
        try:
            test_file = os.path.join(directory, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Проверка записи в {directory}: ОК")
        except Exception as e:
            error_logger.log_exception(
                e,
                "main",
                "permission_check",
                f"ВНИМАНИЕ! Проблема с правами доступа к {directory}"
            )
            error_logger.log_error(
                "Сохранение моделей может не работать!",
                "main",
                "permission_check"
            )
    
    # Загрузка моделей
    print("Запуск сервера. Загрузка сохраненных моделей...")
    try:
        voice_loaded, emotion_loaded = load_models(basedir)
        if voice_loaded and emotion_loaded:
            print("Все модели успешно загружены")
        elif voice_loaded:
            print("Загружена только модель идентификации голоса")
        elif emotion_loaded:
            print("Загружена только модель распознавания эмоций")
        else:
            print("Не удалось загрузить модели")
    except Exception as e:
        error_logger.log_exception(
            e,
            "main",
            "model_loading",
            "Ошибка при загрузке моделей"
        )
        print("Продолжение с новыми моделями")
    
    # Принудительно сохраняем модели прямо перед запуском для тестирования
    # Это позволит проверить, что пути и права доступа правильные
    save_models_on_exit()
    
    # Запуск приложения
    print("Сервер готов. Для завершения нажмите Ctrl+C")
    try:
        # Отключаем режим отладки Flask, чтобы функция atexit работала корректно
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nПолучен сигнал завершения. Сохранение данных...")
        save_models_on_exit()
        print("Завершение работы сервера.")
    except Exception as e:
        error_logger.log_exception(
            e,
            "main",
            "server_run",
            "Критическая ошибка при запуске сервера"
        )
