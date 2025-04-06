import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pickle
import time
from backend.processors.dataset_creator import extract_features
from backend.api.error_logger import error_logger

class EmotionRecognitionModel:
    def __init__(self):
        self.model = None
        self.emotion_labels = ['гнев', 'радость', 'грусть']
        self.is_trained = False
        # Используем абсолютные пути относительно корня проекта
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'emotion_recognition')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
        self._create_model()
    
    def _create_model(self):
        """
        Создание модели TDNN (Time Delay Neural Network) для распознавания эмоций
        """
        # Размер входных данных (примерно, будет адаптирован под реальные данные)
        input_shape = (None, 22)  # 20 MFCC + 2 доп. признака
        
        # Создание модели TDNN
        inputs = layers.Input(shape=input_shape)
        
        # Первый свёрточный слой с расширенной свёрткой
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', dilation_rate=1)(inputs)
        x = layers.BatchNormalization()(x)
        
        # Второй свёрточный слой с расширенной свёрткой
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', dilation_rate=2)(x)
        x = layers.BatchNormalization()(x)
        
        # Третий свёрточный слой с расширенной свёрткой
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', dilation_rate=4)(x)
        x = layers.BatchNormalization()(x)
        
        # Глобальный пулинг
        x = layers.GlobalAveragePooling1D()(x)
        
        # Полносвязные слои
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Выходной слой (3 класса: гнев, радость, грусть)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        # Инициализация модели
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Компиляция модели
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, dataset):
        """
        Обучение или дообучение модели
        """
        # Проверка наличия данных
        if not dataset:
            raise ValueError("Пустой датасет")
        
        # Извлечение признаков и меток из датасета
        features = np.array([item['features'] for item in dataset])
        labels = [item['label'] for item in dataset]
        
        # Проверка, что все метки содержатся в списке допустимых эмоций
        invalid_labels = [label for label in labels if label not in self.emotion_labels]
        if invalid_labels:
            raise ValueError(f"Недопустимые метки эмоций: {invalid_labels}. Допустимые: {self.emotion_labels}")
        
        # Преобразование текстовых меток в числовые
        numeric_labels = np.array([self.emotion_labels.index(label) for label in labels])
        
        # Обучение модели
        self.model.fit(
            features, numeric_labels,
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
        
        self.is_trained = True
    
    def predict(self, audio_fragments):
        """
        Предсказание эмоции по аудиофрагментам
        """
        # Проверка, что модель обучена
        if not self.is_trained:
            return "unknown"
        
        # Проверка на наличие фрагментов
        if not audio_fragments:
            return "unknown"
        
        try:
            # Извлечение признаков из каждого фрагмента
            features_list = []
            for fragment in audio_fragments:
                features = extract_features(fragment)
                features_list.append(features)
            
            # Среднее значение признаков по всем фрагментам
            avg_features = np.mean(features_list, axis=0)
            
            # Изменение формы для подачи в модель
            input_data = np.expand_dims(avg_features, axis=0)
            
            # Предсказание класса
            predictions = self.model.predict(input_data)
            predicted_class = np.argmax(predictions[0])
            
            # Проверка, что predicted_class находится в пределах допустимого диапазона
            if predicted_class < 0 or predicted_class >= len(self.emotion_labels):
                return "unknown"
            
            # Возвращение названия эмоции
            return self.emotion_labels[predicted_class]
        except Exception as e:
            error_message = f"Ошибка при предсказании эмоции: {str(e)}"
            # Логируем ошибку
            error_logger.log_error(error_message, "model", "emotion_recognition")
            
            return "unknown"
    
    def reset(self):
        """
        Сброс модели до начального состояния
        """
        self._create_model()
        self.is_trained = False
    
    def save(self):
        """
        Сохранение модели в файл
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'emotion_model_{timestamp}')
        
        # Сохранение модели TensorFlow
        self.model.save(f'{model_path}.h5')
        
        # Сохранение дополнительных данных
        with open(f'{model_path}_metadata.pkl', 'wb') as f:
            pickle.dump({
                'is_trained': self.is_trained
            }, f)
        
        return model_path
    
    def load(self, path):
        """
        Загрузка модели из файла
        """
        # Загрузка модели TensorFlow
        self.model = models.load_model(f'{path}.h5')
        
        # Загрузка дополнительных данных
        with open(f'{path}_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.is_trained = metadata['is_trained']
