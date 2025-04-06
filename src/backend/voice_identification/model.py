import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils
import pickle
import time
from backend.processors.dataset_creator import extract_features
from backend.api.error_logger import error_logger

class VoiceIdentificationModel:
    def __init__(self):
        self.model = None
        self.user_labels = []
        self.is_trained = False
        # Используем абсолютные пути относительно корня проекта
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'voice_identification')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
        self._create_model()
    
    def _create_model(self):
        """
        Улучшенная архитектура модели для идентификации голоса
        """
        # Увеличиваем размерность для более детальных признаков
        inputs = layers.Input(shape=(None, 134))
        
        # Добавляем больше слоев и улучшаем архитектуру
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)  # Предотвращаем переобучение
        
        # Добавляем глубину сети
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        
        # Добавляем внимание для фокусировки на ключевых частях аудио
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(256)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Применяем внимание
        x = layers.Multiply()([x, attention])
        x = layers.GlobalAveragePooling1D()(x)
        
        # Полносвязные слои
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Выходной слой будет динамически настроен в train()
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
    
    def train(self, dataset):
        """
        Обучение модели на наборе данных
        
        Args:
            dataset: Список словарей с ключами 'features' и 'label'
        """
        if not dataset:
            raise ValueError("Пустой датасет. Невозможно обучить модель.")
        
        # Стандартизация размеров признаков перед обработкой
        max_feature_shape = None
        for item in dataset:
            features = item['features']
            if max_feature_shape is None or features.shape[1] > max_feature_shape:
                max_feature_shape = features.shape[1]
        
        # Приведение всех признаков к одинаковой размерности
        processed_features = []
        labels = []
        
        for item in dataset:
            features = item['features']
            # Если размер меньше максимального, заполним нулями
            if features.shape[1] < max_feature_shape:
                padded = np.zeros((features.shape[0], max_feature_shape))
                padded[:, :features.shape[1]] = features
                processed_features.append(padded)
            else:
                processed_features.append(features)
            
            # Добавление метки
            labels.append(item['label'])
        
        # Преобразование в numpy-массивы
        features = np.array(processed_features)
        
        # Получение уникальных имен пользователей
        unique_labels = sorted(list(set(labels)))
        
        # Перестройка выходного слоя модели для правильного количества классов
        num_classes = len(unique_labels)
        if num_classes > 1:
            # Для мультиклассовой классификации
            model_config = self.model.get_config()
            output_layer_config = model_config['layers'][-1]['config']
            output_layer_config['units'] = num_classes
            output_layer_config['activation'] = 'softmax'
            
            # Создаем новую модель с обновленным выходным слоем
            new_model = models.Model.from_config(model_config)
            # Копируем веса из старой модели в новую (кроме последнего слоя)
            for i, layer in enumerate(self.model.layers[:-1]):
                new_model.layers[i].set_weights(layer.get_weights())
            
            self.model = new_model
        
        # Компиляция модели
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Преобразование текстовых меток в индексы
        label_indices = [unique_labels.index(label) for label in labels]
        
        # Изменяем эту строку: вместо np_utils используем tf.keras.utils
        y = utils.to_categorical(label_indices)
        
        # Создаем callbacks для контроля обучения
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='accuracy',  # Используем 'accuracy' вместо 'val_accuracy', если нет валидационных данных
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',  # Используем 'loss' вместо 'val_loss'
                factor=0.5,
                patience=2
            )
        ]
        
        # Обучение модели - в TensorFlow 2.x история обучения возвращается автоматически
        history = self.model.fit(
            features, y, 
            epochs=20,
            batch_size=max(1, min(32, len(dataset) // 4)),  # Адаптивный размер батча
            callbacks=callbacks,
            verbose=1
        )
        
        # Логируем результаты обучения
        try:
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            error_logger.log_error(
                f"Обучение модели идентификации завершено. Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}",
                "training",
                "voice_identification"
            )
        except (KeyError, IndexError):
            pass
        
        # Сохранение меток пользователей
        self.user_labels = unique_labels
        self.is_trained = True
    
    def predict(self, audio_fragments):
        """
        Предсказание имени пользователя по аудиофрагментам
        """
        # Определяем порог уверенности
        CONFIDENCE_THRESHOLD = 0.6
        
        # Проверка, что модель обучена
        if not self.is_trained or not self.user_labels:
            raise ValueError("Модель не обучена")
        
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
        try:
            avg_features = np.mean(features_list, axis=0)
            
            # Проверка на NaN значения
            if np.isnan(avg_features).any():
                avg_features = np.nan_to_num(avg_features)
        except Exception as e:
            raise ValueError(f"Ошибка при обработке признаков: {str(e)}")
        
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
        max_confidence = predictions[0][predicted_class]
        
        # Проверка индекса класса
        if predicted_class < 0 or predicted_class >= len(self.user_labels):
            return "unknown"
        
        # Проверка уверенности предсказания - это единственное место,
        # где мы возвращаем "unknown" как легитимный результат
        if max_confidence < CONFIDENCE_THRESHOLD:
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
