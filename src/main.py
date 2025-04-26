from flask import Flask, send_from_directory, redirect
from backend.api.routes import api_bp
from backend.api.error_logger import error_logger
import os
import logging

# Абсолютный путь к директории src
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, static_folder=None)  # Убираем стандартную папку static
app.register_blueprint(api_bp, url_prefix='/api')

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
