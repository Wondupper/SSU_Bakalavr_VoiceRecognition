from flask import Flask, send_from_directory, redirect
import os
import logging

# Абсолютный путь к директории src
basedir = os.path.abspath(os.path.dirname(__file__))

from backend.api.routes import api_bp, voice_id_model, emotion_model
from backend.ml.common.data_loader import load_emotions_dataset, load_voice_dataset
from backend.loggers.error_logger import error_logger
from backend.loggers.info_logger import info_logger

app = Flask(__name__, static_folder=None)  # Убираем стандартную папку static
app.register_blueprint(api_bp, url_prefix='/api')

# Маршруты для фронтенда
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # Главная страница
    if path == "":
        return send_from_directory(os.path.join(basedir, 'frontend/identification'), 'index.html')
    
    # Перенаправление устаревшего пути /common.js на новый
    if path == "common.js":
        return redirect("/common/common.js")
    
    # CSS и JavaScript файлы для разных разделов
    if path.endswith('.css') or path.endswith('.js'):
        directory, filename = os.path.split(path)
        return send_from_directory(os.path.join(basedir, 'frontend', directory), filename)
    
    # Для всех остальных путей возвращаем главную страницу
    return send_from_directory(os.path.join(basedir, 'frontend/identification'), 'index.html')

@app.route('/common/<path:filename>')
def serve_common_files(filename):
    return send_from_directory(os.path.join(basedir, 'frontend/common'), filename)

def initialize_models():
    """
    Загружает данные и обучает модели при старте приложения
    """
    info_logger.info("Начало инициализации моделей")
    
    # Загрузка наборов данных
    voice_dataset = load_voice_dataset()
    emotions_dataset = load_emotions_dataset()
    # Обучение модели идентификации голоса
    voice_success = voice_id_model.train(voice_dataset)
    # Обучение модели распознавания эмоций
    emotions_success = emotion_model.train(emotions_dataset)
    
    return voice_success and emotions_success

if __name__ == '__main__':
    # Настройка логирования для Flask
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Инициализация и обучение моделей
    models_initialized = initialize_models()
    if not models_initialized:
        error_logger.log_error(
            "Не удалось инициализировать все модели. Приложение может работать некорректно.",
            "main",
            "server_run"
        )
    
    # Запуск приложения
    print("Сервер готов. Для завершения нажмите Ctrl+C")
    try:
        # Запускаем сервер
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nПолучен сигнал завершения.")
        print("Завершение работы сервера.")
    except Exception as e:
        error_logger.log_exception(
            e,
            "main",
            "server_run",
            "Критическая ошибка при запуске сервера"
        )
