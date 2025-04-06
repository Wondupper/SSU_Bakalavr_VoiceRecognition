from flask import Flask, send_from_directory, redirect
from backend.api.routes import api_bp, voice_id_model, emotion_model  # Импортируем модели
import os
import signal
import atexit
import logging
import glob

# Абсолютный путь к директории src
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, static_folder=None)  # Убираем стандартную папку static
app.register_blueprint(api_bp, url_prefix='/api')

# Функция для загрузки сохраненных моделей
def load_saved_models():
    """
    Загрузка последних сохранённых моделей (если они есть)
    """
    try:
        # Загрузка модели идентификации голоса
        voice_models = glob.glob(os.path.join(basedir, 'models', 'voice_identification', 'voice_id_model_*_metadata.pkl'))
        if voice_models:
            # Сортируем по времени создания (убывание)
            voice_models.sort(reverse=True)
            # Берем самую новую модель
            latest_model = voice_models[0]
            # Путь к модели без расширения
            model_path = latest_model.rsplit('_metadata.pkl', 1)[0]
            try:
                voice_id_model.load(model_path)
                print(f"Загружена модель идентификации голоса из {model_path}")
            except Exception as e:
                print(f"Ошибка загрузки модели идентификации голоса: {str(e)}")
        else:
            print("Не найдены сохраненные модели идентификации голоса")
    except Exception as e:
        print(f"Ошибка при поиске моделей идентификации голоса: {str(e)}")
    
    try:
        # Загрузка модели распознавания эмоций
        emotion_models = glob.glob(os.path.join(basedir, 'models', 'emotion_recognition', 'emotion_model_*_metadata.pkl'))
        if emotion_models:
            # Сортируем по времени создания (убывание)
            emotion_models.sort(reverse=True)
            # Берем самую новую модель
            latest_model = emotion_models[0]
            # Путь к модели без расширения
            model_path = latest_model.rsplit('_metadata.pkl', 1)[0]
            try:
                emotion_model.load(model_path)
                print(f"Загружена модель распознавания эмоций из {model_path}")
            except Exception as e:
                print(f"Ошибка загрузки модели распознавания эмоций: {str(e)}")
        else:
            print("Не найдены сохраненные модели распознавания эмоций")
    except Exception as e:
        print(f"Ошибка при поиске моделей распознавания эмоций: {str(e)}")

# Функция для сохранения моделей при завершении работы
def save_models_on_exit():
    print("Сохранение моделей перед завершением работы...")
    
    try:
        if voice_id_model.is_trained:
            voice_id_model.save()
            print("Модель идентификации голоса сохранена")
    except Exception as e:
        print(f"Ошибка при сохранении модели идентификации голоса: {str(e)}")
    
    try:
        if emotion_model.is_trained:
            emotion_model.save()
            print("Модель распознавания эмоций сохранена")
    except Exception as e:
        print(f"Ошибка при сохранении модели распознавания эмоций: {str(e)}")
    
    print("Процесс сохранения завершен")

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
    if path in ['identification', 'training', 'emtraining', 'panel']:
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
    models_dir = os.path.join(basedir, 'models')
    voice_models_dir = os.path.join(models_dir, 'voice_identification')
    emotion_models_dir = os.path.join(models_dir, 'emotion_recognition')

    for directory in [models_dir, voice_models_dir, emotion_models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Создана директория: {directory}")
    
    # Загрузка моделей
    print("Запуск сервера. Загрузка сохраненных моделей...")
    try:
        load_saved_models()
        print("Модели успешно загружены")
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {str(e)}")
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
