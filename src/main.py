from flask import Flask, send_from_directory
from backend.api.routes import api_bp
import os

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
    
    # CSS и JavaScript файлы для разных разделов
    if path.endswith('.css') or path.endswith('.js'):
        directory, filename = os.path.split(path)
        return send_from_directory(os.path.join(basedir, 'frontend', directory), filename)
    
    # Специальные разделы сайта
    if path in ['identification', 'training', 'emtraining', 'panel']:
        return send_from_directory(os.path.join(basedir, f'frontend/{path}'), 'index.html')
    
    # Для всех остальных путей возвращаем главную страницу
    return send_from_directory(os.path.join(basedir, 'frontend/home'), 'index.html')

@app.route('/common.js')
def serve_common_js():
    return send_from_directory(os.path.join(basedir, 'frontend'), 'common.js')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
