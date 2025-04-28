import numpy as np
import tensorflow as tf
from backend.api.error_logger import error_logger
from backend.config import EMOTIONS

class EmotionRecognitionModel:
    """
    Модель для распознавания эмоций в речи.
    Использует архитектуру TDNN (Time Delay Neural Network).
    
    Атрибуты:
        model: Модель TensorFlow для распознавания эмоций
        is_trained: Флаг, указывающий, обучена ли модель
        is_training: Флаг, указывающий, идет ли процесс обучения
    """
    
    def __init__(self):
        """
        Инициализирует модель для распознавания эмоций в речи.
        """
        # Инициализируем атрибуты
        self.model = None
        self.is_trained = False
        self.is_training = False
        
        
    def build_model(self, input_shape, num_classes):
        """
        Создает новую модель TDNN с заданными параметрами.
        
        Args:
            input_shape: Форма входных данных (time_steps, features)
            num_classes: Количество классов (эмоций) для распознавания
            
        Returns:
            Скомпилированная модель TensorFlow
        """
        try:
            # Определяем входной слой
            inputs = tf.keras.layers.Input(shape=input_shape)
            
            # Добавляем слой для обработки временных данных (имитация TDNN)
            # Conv1D с диалатацией для анализа разных временных масштабов
            x = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                dilation_rate=1,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 регуляризация
            )(inputs)
            
            # Добавляем слой с большей диалатацией для захвата долгосрочных зависимостей
            x = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                dilation_rate=2,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 регуляризация
            )(x)
            
            # Добавляем слой пространственного дропаута для лучшей регуляризации
            x = tf.keras.layers.SpatialDropout1D(0.2)(x)
            
            # Добавляем еще один слой TDNN с большей диалатацией
            x = tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                dilation_rate=4,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 регуляризация
            )(x)
            
            # Свертка с шагом для уменьшения размерности
            x = tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 регуляризация
            )(x)
            
            # Добавляем слой нормализации и дропаут для регуляризации
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Еще один сверточный слой для извлечения высокоуровневых признаков
            x = tf.keras.layers.Conv1D(
                filters=256,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 регуляризация
            )(x)
            
            # Глобальный пулинг для получения фиксированного размера вектора признаков
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Полносвязный слой для классификации
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            # Выходной слой с softmax-активацией для многоклассовой классификации
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            # Создаем модель
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            
            # Компилируем модель с оптимизированными параметрами для предотвращения переобучения
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # Уменьшенная скорость обучения
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "emotion_recognition",
                "build_model",
                "Ошибка при построении модели"
            )
            return None
                
    def train(self, features, labels):
        """
        Обучает модель на наборе признаков и соответствующих меток.
        
        Args:
            features: Список признаков (features) для обучения
            labels: Список меток (эмоций) для каждого набора признаков
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        try:
            # Проверка входных данных
            if not features or not labels:
                error_logger.log_error(
                    "Пустые входные данные для обучения",
                    "emotion_recognition",
                    "train"
                )
                return False
                
            if len(features) != len(labels):
                error_logger.log_error(
                    "Количество наборов признаков не соответствует количеству меток",
                    "emotion_recognition",
                    "train"
                )
                return False
                
            # Проверяем, что все метки являются допустимыми эмоциями
            for label in labels:
                if label not in EMOTIONS:
                    error_logger.log_error(
                        f"Недопустимая метка эмоции: {label}",
                        "emotion_recognition",
                        "train"
                    )
                    return False
                    
            # Устанавливаем флаг, что идет обучение
            self.is_training = True
            
            # Создаем отображение меток эмоций на числовые индексы
            label_to_index = {emotion: i for i, emotion in enumerate(EMOTIONS)}
            
            # Преобразуем текстовые метки в числовые индексы
            numeric_labels = np.array([label_to_index[label] for label in labels])
            
            # Проверяем, есть ли признаки
            if not features:
                error_logger.log_error(
                    "Не предоставлены признаки для обучения",
                    "emotion_recognition",
                    "train"
                )
                self.is_training = False
                return False
                
            # Преобразуем список в numpy массив
            X = np.array(features)
            
            # Получаем размерность входных данных для модели
            input_shape = (X.shape[1], X.shape[2])
            
            # Создаем новую модель
            if self.model is None:
                self.model = self.build_model(input_shape, len(EMOTIONS))
                
                if self.model is None:
                    error_logger.log_error(
                        "Не удалось создать модель",
                        "emotion_recognition",
                        "train"
                    )
                    self.is_training = False
                    return False
                    
            # Обучаем модель с улучшенными параметрами для предотвращения переобучения
            history = self.model.fit(
                X, numeric_labels,
                epochs=100,  # Увеличиваем число эпох
                batch_size=min(16, len(X)//2 + 1),  # Меньший размер батча
                validation_split=0.3,  # Увеличиваем долю валидационной выборки
                verbose=1,
                class_weight=self._get_class_weights(numeric_labels),  # Добавляем веса классов
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',  # Используем val_loss вместо val_accuracy
                        patience=20,  # Увеличиваем терпение
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=0.00001
                    )
                ]
            )
            
            # Устанавливаем флаг, что модель обучена
            self.is_trained = True
            
            # Логируем результаты обучения
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            
            # Сбрасываем флаг обучения
            self.is_training = False
            
            return True
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "emotion_recognition",
                "train",
                "Ошибка при обучении модели"
            )
            # Сбрасываем флаг обучения в случае ошибки
            self.is_training = False
            return False
            
    def predict(self, features_list):
        """
        Распознает эмоцию из признаков.
        
        Args:
            features_list: Список признаков для распознавания
            
        Returns:
            list: Список результатов распознавания для каждого набора признаков
        """
        try:
            # Проверка состояния модели
            if not self.is_trained or self.model is None:
                error_logger.log_error(
                    "Модель не обучена или не инициализирована",
                    "emotion_recognition",
                    "predict"
                )
                return []
                
            # Проверка входных данных
            if not features_list or len(features_list) == 0:
                error_logger.log_error(
                    "Пустой список признаков",
                    "emotion_recognition",
                    "predict"
                )
                return []
                
            # Преобразуем список в numpy массив
            X = np.array(features_list)
            
            # Получаем предсказания модели для всех наборов признаков
            predictions = self.model.predict(X)
            
            # Формируем результаты для каждого набора признаков
            results = []
            for i, prediction in enumerate(predictions):
                # Находим класс с наибольшей вероятностью
                predicted_class_index = np.argmax(prediction)
                confidence = prediction[predicted_class_index]
                
                # Если уверенность выше порога, распознаем эмоцию
                if confidence >= 0.8:
                    label = EMOTIONS[predicted_class_index]
                else:
                    label = "unknown"
                
                # Добавляем результат в список
                results.append({
                    'label': label,
                    'confidence': float(confidence)
                })
            
            return results
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "emotion_recognition",
                "predict",
                "Ошибка при распознавании эмоции"
            )
            return []
            
    def _get_class_weights(self, labels):
        """
        Вычисляет веса классов для несбалансированных данных
        
        Args:
            labels: Числовые метки классов
            
        Returns:
            dict: Словарь весов классов
        """
        try:
            from sklearn.utils.class_weight import compute_class_weight
            import numpy as np
            
            classes = np.unique(labels)
            if len(classes) > 1:
                weights = compute_class_weight(class_weight='balanced', 
                                              classes=classes, 
                                              y=labels)
                return {i: w for i, w in zip(classes, weights)}
            else:
                return None
        except Exception as e:
            error_logger.log_exception(
                e,
                "emotion_recognition",
                "_get_class_weights",
                "Ошибка при вычислении весов классов"
            )
            return None
            
