import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pickle
import json

class VoiceIdentificationModel:
    def __init__(self):
        self.model = None
        self.user_labels = []
        self.is_trained = False
        self.model_dir = os.path.join('..', 'models', 'voice_identification')
        self._create_model()
    
    def _create_model(self):
        """
        Создание модели TDNN (Time Delay Neural Network) для идентификации по голосу
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
        
        # Выходной слой (будет меняться в зависимости от количества пользователей)
        outputs = layers.Dense(1, activation='softmax')(x)
        
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
        
        # Обновление списка пользователей
        for label in labels:
            if label not in self.user_labels:
                self.user_labels.append(label)
        
        # Преобразование текстовых меток в числовые
        numeric_labels = np.array([self.user_labels.index(label) for label in labels])
        
        # Проверка и пересоздание выходного слоя, если количество пользователей изменилось
        if self.model.layers[-1].units != len(self.user_labels):
            # Сохранение весов существующих слоев
            weights = [layer.get_weights() for layer in self.model.layers[:-1]]
            
            # Пересоздание модели с новым количеством классов
            inputs = self.model.inputs
            x = inputs[0]
            
            for i, layer in enumerate(self.model.layers[1:-1]):
                x = layer(x)
            
            # Новый выходной слой
            outputs = layers.Dense(len(self.user_labels), activation='softmax')(x)
            
            # Создание новой модели
            self.model = models.Model(inputs=inputs, outputs=outputs)
            
            # Восстановление весов
            for i, layer in enumerate(self.model.layers[1:-1]):
                layer.set_weights(weights[i])
            
            # Перекомпиляция модели
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
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
        Предсказание идентификации пользователя по аудиофрагментам
        """
        if not self.is_trained:
            return "unknown"
        
        if not audio_fragments:
            return "unknown"
        
        # Извлечение признаков из аудиофрагментов
        features = np.array([extract_features(fragment) for fragment in audio_fragments])
        
        # Получение предсказаний для всех фрагментов
        predictions = self.model.predict(features)
        
        # Усреднение предсказаний
        avg_prediction = np.mean(predictions, axis=0)
        
        # Определение наиболее вероятного класса
        predicted_class = np.argmax(avg_prediction)
        
        # Если вероятность ниже порога, вернуть "unknown"
        if avg_prediction[predicted_class] < 0.7:
            return "unknown"
        
        # Возвращение имени пользователя
        return self.user_labels[predicted_class]
    
    def reset(self):
        """
        Сброс модели до начального состояния
        """
        self._create_model()
        self.user_labels = []
        self.is_trained = False
    
    def save(self):
        """
        Сохранение модели в файл
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'voice_id_model_{timestamp}')
        
        # Сохранение модели TensorFlow
        self.model.save(f'{model_path}.h5')
        
        # Сохранение дополнительных данных
        with open(f'{model_path}_metadata.pkl', 'wb') as f:
            pickle.dump({
                'user_labels': self.user_labels,
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
            self.user_labels = metadata['user_labels']
            self.is_trained = metadata['is_trained']
