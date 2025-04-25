import os
import tensorflow as tf
import glob
from backend.api.error_logger import error_logger
from backend.config import VOICE_ID_MODELS_DIR, EMOTION_MODELS_DIR
import time


def save_model(model, is_trained, filepath):
    """
    Сохраняет модель в файл.
    
    Args:
        model: Модель TensorFlow
        is_trained: Флаг, указывающий, обучена ли модель
        filepath: Путь к файлу для сохранения модели
        
    Returns:
        bool: Успешно ли сохранена модель
    """
    try:
        # Проверка состояния модели
        if not is_trained or model is None:
            return False
            
        # Создаем директорию для модели, если она не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Сохраняем модель
        model.save(filepath)
        
        return True
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "model_loader_or_saver",
            "save_model",
            "Ошибка при сохранении модели"
        )
        return False
        
def load_model(filepath):
    """
    Загружает модель из файла.
    
    Args:
        filepath: Путь к файлу с сохраненной моделью
        
    Returns:
        tuple: (model, is_trained, success)
            model: Загруженная модель или None в случае ошибки
            is_trained: Флаг, указывающий, обучена ли модель
            success: Успешно ли загружена модель
    """
    try:
        # Проверяем существование файла модели
        if not os.path.exists(filepath):
            error_logger.log_error(
                f"Файл модели не найден: {filepath}",
                "model_loader_or_saver",
                "load_model"
            )
            return None, False, False
            
        # Загружаем модель
        model = tf.keras.models.load_model(filepath)
        
        return model, True, True
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "model_loader_or_saver",
            "load_model",
            "Ошибка при загрузке модели"
        )
        # Возвращаем None и флаги в случае ошибки
        return None, False, False


def load_models(voice_id_model, emotion_model, basedir):
    """
    Загрузка последних сохранённых моделей (если они есть)
    
    Args:
        voice_id_model: Экземпляр модели идентификации голоса
        emotion_model: Экземпляр модели распознавания эмоций
        basedir: Базовая директория проекта
        
    Returns:
        tuple: (voice_id_loaded, emotion_loaded) 
            voice_id_loaded: Была ли загружена модель идентификации
            emotion_loaded: Была ли загружена модель эмоций
    """
    voice_id_loaded = False
    emotion_loaded = False
    
    try:
        # Загрузка модели идентификации голоса
        voice_models = glob.glob(os.path.join(basedir, VOICE_ID_MODELS_DIR, '*.*'))
        voice_model_files = [f for f in voice_models if f.endswith('.h5')]
        
        if voice_model_files:
            # Сортируем по времени создания (убывание)
            voice_model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            # Берем самую новую модель
            latest_model = voice_model_files[0]
            # Путь к модели без расширения
            model_path = latest_model.rsplit('.h5', 1)[0]
            try:
                # Загружаем модель
                model, is_trained, success = load_model(model_path)
                if success:
                    voice_id_model.model = model
                    voice_id_model.is_trained = is_trained
                    
                    # Пытаемся загрузить классы, если они есть
                    classes_file = os.path.join(os.path.dirname(model_path), "classes.txt")
                    if os.path.exists(classes_file):
                        voice_id_model.classes = []
                        with open(classes_file, 'r') as f:
                            for line in f:
                                voice_id_model.classes.append(line.strip())
                    
                    print(f"Загружена модель идентификации голоса из {model_path}")
                    voice_id_loaded = True
                else:
                    print(f"Не удалось загрузить модель идентификации голоса из {model_path}")
            except Exception as e:
                error_logger.log_exception(
                    e,
                    "model_loader_or_saver",
                    "load_models",
                    f"Ошибка загрузки модели идентификации голоса: {str(e)}"
                )
                print(f"Ошибка загрузки модели идентификации голоса: {str(e)}")
        else:
            print("Не найдены сохраненные модели идентификации голоса")
    except Exception as e:
        error_logger.log_exception(
            e,
            "model_loader_or_saver",
            "load_models",
            f"Ошибка при поиске моделей идентификации голоса: {str(e)}"
        )
        print(f"Ошибка при поиске моделей идентификации голоса: {str(e)}")
    
    try:
        # Загрузка модели распознавания эмоций
        emotion_models = glob.glob(os.path.join(basedir, EMOTION_MODELS_DIR, '*.*'))
        emotion_model_files = [f for f in emotion_models if f.endswith('.h5')]
        
        if emotion_model_files:
            # Сортируем по времени создания (убывание)
            emotion_model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            # Берем самую новую модель
            latest_model = emotion_model_files[0]
            # Путь к модели без расширения
            model_path = latest_model.rsplit('.h5', 1)[0]
            try:
                # Загружаем модель
                model, is_trained, success = load_model(model_path)
                if success:
                    emotion_model.model = model
                    emotion_model.is_trained = is_trained
                    print(f"Загружена модель распознавания эмоций из {model_path}")
                    emotion_loaded = True
                else:
                    print(f"Не удалось загрузить модель распознавания эмоций из {model_path}")
            except Exception as e:
                error_logger.log_exception(
                    e,
                    "model_loader_or_saver",
                    "load_models",
                    f"Ошибка загрузки модели распознавания эмоций: {str(e)}"
                )
                print(f"Ошибка загрузки модели распознавания эмоций: {str(e)}")
        else:
            print("Не найдены сохраненные модели распознавания эмоций")
    except Exception as e:
        error_logger.log_exception(
            e,
            "model_loader_or_saver",
            "load_models",
            f"Ошибка при поиске моделей распознавания эмоций: {str(e)}"
        )
        print(f"Ошибка при поиске моделей распознавания эмоций: {str(e)}")
        
    return voice_id_loaded, emotion_loaded

def save_models(voice_id_model, emotion_model, basedir):
    """
    Сохранение моделей
    
    Args:
        voice_id_model: Экземпляр модели идентификации голоса
        emotion_model: Экземпляр модели распознавания эмоций
        basedir: Базовая директория проекта
        
    Returns:
        tuple: (voice_id_path, emotion_path) 
            voice_id_path: Путь к сохраненной модели идентификации или None
            emotion_path: Путь к сохраненной модели эмоций или None
    """
    voice_id_path = None
    emotion_path = None
    
    print("Сохранение моделей...")
    
    # Сохранение модели идентификации голоса
    try:
        if voice_id_model.is_trained and voice_id_model.model is not None:
            # Создаем директорию для сохранения
            save_dir = os.path.join(basedir, VOICE_ID_MODELS_DIR)
            os.makedirs(save_dir, exist_ok=True)
            
            # Генерируем имя файла с текущим временем
            timestamp = int(time.time())
            filepath = os.path.join(save_dir, f"voice_id_model_{timestamp}")
            
            # Сохраняем модель
            if save_model(voice_id_model.model, voice_id_model.is_trained, filepath):
                # Сохраняем список классов в отдельный файл
                classes_file = os.path.join(os.path.dirname(filepath), "classes.txt")
                try:
                    with open(classes_file, 'w') as f:
                        for cls in voice_id_model.classes:
                            f.write(f"{cls}\n")
                except Exception as e:
                    error_logger.log_exception(
                        e,
                        "model_loader_or_saver",
                        "save_models",
                        f"Ошибка при сохранении классов модели идентификации: {str(e)}"
                    )
                
                voice_id_path = filepath
                print(f"Модель идентификации голоса сохранена в {filepath}")
            else:
                print("Не удалось сохранить модель идентификации голоса")
        else:
            print("Модель идентификации голоса не обучена, сохранение не выполнено")
    except Exception as e:
        error_logger.log_exception(
            e,
            "model_loader_or_saver",
            "save_models",
            f"Ошибка при сохранении модели идентификации голоса: {str(e)}"
        )
        print(f"Ошибка при сохранении модели идентификации голоса: {str(e)}")
    
    # Сохранение модели распознавания эмоций
    try:
        # Принудительно выводим информацию о состоянии модели для диагностики
        print(f"Состояние модели эмоций - is_trained: {emotion_model.is_trained}")
        
        if emotion_model.is_trained and emotion_model.model is not None:
            # Создаем директорию для сохранения
            save_dir = os.path.join(basedir, EMOTION_MODELS_DIR)
            os.makedirs(save_dir, exist_ok=True)
            
            # Генерируем имя файла с текущим временем
            timestamp = int(time.time())
            filepath = os.path.join(save_dir, f"emotion_model_{timestamp}")
            
            # Сохраняем модель
            if save_model(emotion_model.model, emotion_model.is_trained, filepath):
                emotion_path = filepath
                print(f"Модель распознавания эмоций сохранена в {filepath}")
            else:
                print("Не удалось сохранить модель распознавания эмоций")
        else:
            print("Модель распознавания эмоций не обучена, сохранение не выполнено")
    except Exception as e:
        error_logger.log_exception(
            e,
            "model_loader_or_saver",
            "save_models",
            f"Ошибка при сохранении модели распознавания эмоций: {str(e)}"
        )
        print(f"Ошибка при сохранении модели распознавания эмоций: {str(e)}")
    
    print("Процесс сохранения завершен")
    
    return voice_id_path, emotion_path 