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
        Создание модели TDNN для распознавания эмоций
        """
        # Исправление размерности входных данных
        input_shape = (None, 134)  # Корректируем размерность
        inputs = layers.Input(shape=input_shape)
        
        # Последовательные слои свертки с расширенной дилатацией
        # Это позволит захватить более широкий контекст
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', dilation_rate=1)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', dilation_rate=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', dilation_rate=4)(x)
        x = layers.BatchNormalization()(x)
        
        # Добавляем параллельные пути обработки для разных временных масштабов
        y = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
        y = layers.BatchNormalization()(y)
        y = layers.MaxPooling1D(pool_size=2)(y)
        
        # Объединяем пути
        x = layers.Concatenate()([x, y])
        
        # Глобальный пулинг
        x = layers.GlobalAveragePooling1D()(x)
        
        # Полносвязные слои
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Выходной слой (3 класса: гнев, радость, грусть)
        outputs = layers.Dense(len(self.emotion_labels), activation='softmax')(x)
        
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

        # Проверка меток эмоций
        labels = [item['label'] for item in dataset]
        invalid_labels = [label for label in labels if label not in self.emotion_labels]
        if invalid_labels:
            raise ValueError(f"Недопустимые метки эмоций: {invalid_labels}. Допустимые: {self.emotion_labels}")

        # Стандартизация размеров признаков перед обработкой
        max_feature_shape = None
        for item in dataset:
            features = item['features']
            if max_feature_shape is None or features.shape[1] > max_feature_shape:
                max_feature_shape = features.shape[1]
        
        # Приведение всех признаков к одинаковой размерности
        processed_features = []
        
        for item in dataset:
            features = item['features']
            # Если размер меньше максимального, заполним нулями
            if features.shape[1] < max_feature_shape:
                padded = np.zeros((features.shape[0], max_feature_shape))
                padded[:, :features.shape[1]] = features
                processed_features.append(padded)
            else:
                processed_features.append(features)
        
        # Преобразование в numpy-массивы
        features = np.array(processed_features)
        
        # Преобразование текстовых меток в числовые
        numeric_labels = np.array([self.emotion_labels.index(label) for label in labels])
        
        # Создаем callbacks для мониторинга обучения
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2
            )
        ]
        
        # Обучение модели с адаптивным размером батча
        history = self.model.fit(
            features, numeric_labels,
            epochs=20,
            batch_size=max(1, min(32, len(dataset) // 4)),
            callbacks=callbacks,
            verbose=1
        )
        
        # Логируем результаты обучения
        try:
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            error_logger.log_error(
                f"Обучение модели эмоций завершено. Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}",
                "training",
                "emotion_recognition"
            )
        except (KeyError, IndexError):
            pass
        
        self.is_trained = True
    
    def predict(self, audio_fragments):
        """
        Предсказание эмоции по аудиофрагментам
        """
        # Проверка, что модель обучена
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Проверка, что метки эмоций определены
        if not self.emotion_labels:
            self.emotion_labels = ['гнев', 'радость', 'грусть']  # Устанавливаем дефолтные значения
        
        # Проверка на наличие фрагментов
        if not audio_fragments or len(audio_fragments) == 0:
            raise ValueError("Отсутствуют аудиофрагменты для анализа")
        
        # Извлечение признаков из каждого фрагмента
        features_list = []
        for fragment in audio_fragments:
            features = extract_features(fragment)
            features_list.append(features)
        
        # Проверка на наличие признаков после извлечения
        if not features_list or len(features_list) == 0:
            raise ValueError("Не удалось извлечь признаки из аудиофрагментов")
        
        # Среднее значение признаков по всем фрагментам
        avg_features = np.mean(features_list, axis=0)
        
        # Проверка размерности
        if len(avg_features.shape) != 2:
            raise ValueError(f"Неверная размерность признаков: {avg_features.shape}")
            
        # Проверка и исправление размерности признаков
        expected_shape = (None, 134)
        if avg_features.shape[1] != expected_shape[1]:
            if avg_features.shape[1] < expected_shape[1]:
                # Если признаков меньше, дополняем нулями
                padded = np.zeros((avg_features.shape[0], expected_shape[1]))
                padded[:, :avg_features.shape[1]] = avg_features
                avg_features = padded
            else:
                # Если признаков больше, обрезаем
                avg_features = avg_features[:, :expected_shape[1]]
        
        # Подготовка входных данных для модели
        input_data = np.expand_dims(avg_features, axis=0)
        
        # Предсказание класса
        try:
            predictions = self.model.predict(input_data, verbose=0)
        except TypeError:
            predictions = self.model.predict(input_data)
        
        predicted_class = np.argmax(predictions[0])
        
        # Проверка индекса класса
        if predicted_class < 0 or predicted_class >= len(self.emotion_labels):
            # Если индекс за пределами допустимых значений, возвращаем дефолтную эмоцию
            return self.emotion_labels[0]  # Возвращаем первую эмоцию как дефолтную
        
        # Возвращение названия эмоции - всегда возвращаем одну из эмоций
        return self.emotion_labels[predicted_class]
    
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
            os.makedirs(self.model_dir, exist_ok=True)
        
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'emotion_model_{timestamp}')
        
        # Сохранение модели TensorFlow
        self.model.save(f'{model_path}.h5')
        
        # Сохранение дополнительных данных
        with open(f'{model_path}_metadata.pkl', 'wb') as f:
            pickle.dump({
                'is_trained': self.is_trained,
                'emotion_labels': self.emotion_labels  # Добавляем сохранение меток
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
            # Загружаем метки эмоций, если они есть
            if 'emotion_labels' in metadata:
                self.emotion_labels = metadata['emotion_labels']
