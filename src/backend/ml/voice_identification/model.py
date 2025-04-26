import numpy as np
import tensorflow as tf
from backend.api.error_logger import error_logger

class VoiceIdentificationModel:
    """
    Модель для идентификации пользователя по голосу.
    Использует архитектуру TDNN (Time Delay Neural Network).
    
    Атрибуты:
        model: Модель TensorFlow для идентификации по голосу
        classes: Список имен пользователей, которые модель может распознать
        is_trained: Флаг, указывающий, обучена ли модель
        is_training: Флаг, указывающий, идет ли процесс обучения
    """
    
    def __init__(self):
        """
        Инициализирует модель для идентификации по голосу.
        """
        # Инициализируем атрибуты
        self.model = None
        self.classes = []
        self.is_trained = False
        self.is_training = False
        
    def reset_model(self):
        """
        Сбрасывает модель, удаляя все веса и обученную информацию.
        """
        # Удаляем текущую модель из памяти, если она существует
        if self.model is not None:
            self.model = None
            
        # Очищаем список классов
        self.classes = []
        
        # Сбрасываем флаги состояния
        self.is_trained = False
        
    def build_model(self, input_shape, num_classes):
        """
        Создает новую модель TDNN с заданными параметрами.
        
        Args:
            input_shape: Форма входных данных (time_steps, features)
            num_classes: Количество классов (пользователей) для распознавания
            
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
                dilation_rate=1
            )(inputs)
            
            # Добавляем слой с большей диалатацией для захвата долгосрочных зависимостей
            x = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                dilation_rate=2
            )(x)
            
            # Добавляем еще один слой TDNN с большей диалатацией
            x = tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                dilation_rate=4
            )(x)
            
            # Свертка с шагом для уменьшения размерности
            x = tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu'
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
                activation='relu'
            )(x)
            
            # Глобальный пулинг для получения фиксированного размера вектора признаков
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Улучшенный блок представления с использованием остаточных связей
            x_res = x
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            
            # Добавляем остаточную связь для лучшего градиентного потока
            if x_res.shape[-1] == 128:
                x = x + x_res
            else:
                x_res = tf.keras.layers.Dense(128, activation=None)(x_res)
                x = x + x_res
                
            # Нормализация перед выходным слоем
            x = tf.keras.layers.BatchNormalization()(x)
            
            # Выходной слой с softmax-активацией для многоклассовой классификации
            # Важно: проверяем, что num_classes > 1 для избежания ошибки softmax
            if num_classes > 1:
                outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            else:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            # Создаем модель
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            
            # Компилируем модель с более подходящими параметрами для маленьких выборок
            if num_classes > 1:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
            
            return model
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "build_model",
                "Ошибка при построении модели"
            )
            return None
                
    def train(self, features, labels):
        """
        Обучает модель на наборе признаков и соответствующих меток.
        
        Args:
            features: Список признаков (features) для обучения
            labels: Список меток (имена пользователей) для каждого набора признаков
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        try:
            # Проверка входных данных
            if not features or not labels:
                error_logger.log_error(
                    "Пустые входные данные для обучения",
                    "voice_identification",
                    "train"
                )
                return False
                
            if len(features) != len(labels):
                error_logger.log_error(
                    "Количество наборов признаков не соответствует количеству меток",
                    "voice_identification",
                    "train"
                )
                return False
                
            # Устанавливаем флаг, что идет обучение
            self.is_training = True
            
            # Обновляем список классов
            unique_labels = sorted(list(set(labels)))
            
            # Проверяем, создана ли модель и соответствует ли она текущим классам
            if self.model is None or len(self.classes) != len(unique_labels) or not all(x in self.classes for x in unique_labels):
                # Обновляем список классов
                self.classes = unique_labels
                
                # Создаем отображение имен на индексы
                label_to_index = {label: i for i, label in enumerate(self.classes)}
                
                # Преобразуем текстовые метки в числовые индексы
                numeric_labels = np.array([label_to_index[label] for label in labels])
                
                # Проверяем, есть ли признаки
                if not features:
                    error_logger.log_error(
                        "Не предоставлены признаки для обучения",
                        "voice_identification",
                        "train"
                    )
                    self.is_training = False
                    return False
                    
                # Преобразуем список в numpy массив
                X = np.array(features)
                
                # Получаем размерность входных данных для модели
                input_shape = (X.shape[1], X.shape[2])
                
                # Создаем новую модель
                self.model = self.build_model(input_shape, len(self.classes))
                
                if self.model is None:
                    error_logger.log_error(
                        "Не удалось создать модель",
                        "voice_identification",
                        "train"
                    )
                    self.is_training = False
                    return False
                
                # Улучшения для малых выборок: аугментация на лету
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.GaussianNoise(0.1),
                ])
                
                # Обучаем модель с нуля с улучшенными параметрами
                history = self.model.fit(
                    data_augmentation(X, training=True), numeric_labels,
                    epochs=100,  # Увеличиваем число эпох
                    batch_size=min(32, len(X)//2 + 1),  # Адаптируем размер батча
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=15,  # Увеличиваем терпение для маленьких выборок
                            restore_best_weights=True
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=5,
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
            
            else:
                # Модель уже существует, выполняем дообучение
                
                # Создаем отображение имен на индексы
                label_to_index = {label: i for i, label in enumerate(self.classes)}
                
                # Преобразуем текстовые метки в числовые индексы
                numeric_labels = np.array([label_to_index[label] for label in labels])
                
                # Проверяем, есть ли признаки
                if not features:
                    error_logger.log_error(
                        "Не предоставлены признаки для дообучения",
                        "voice_identification",
                        "train"
                    )
                    self.is_training = False
                    return False
                    
                # Преобразуем список в numpy массив
                X = np.array(features)
                
                # Аугментация данных для дообучения
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.GaussianNoise(0.1),
                ])
                
                # Дообучаем существующую модель с улучшенными параметрами
                history = self.model.fit(
                    data_augmentation(X, training=True), numeric_labels,
                    epochs=50,  # Умеренное количество эпох для дообучения
                    batch_size=min(32, len(X)//2 + 1),  # Адаптивный размер батча
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=10,  # Увеличенное терпение
                            restore_best_weights=True
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=3,
                            min_lr=0.00001
                        )
                    ]
                )
                
                # Устанавливаем флаг, что модель обучена
                self.is_trained = True
                
                # Логируем результаты дообучения
                final_loss = history.history['loss'][-1]
                final_accuracy = history.history['accuracy'][-1]
                
                # Сбрасываем флаг обучения
                self.is_training = False
                
                return True
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "train",
                "Ошибка при обучении модели"
            )
            
            # Сбрасываем флаг обучения в случае ошибки
            self.is_training = False
            return False
            
    def predict(self, features_list):
        """
        Идентифицирует пользователя по голосу из признаков.
        
        Args:
            features_list: Список признаков для идентификации
            
        Returns:
            list: Список результатов распознавания для каждого набора признаков
        """
        try:
            # Проверка состояния модели
            if not self.is_trained or self.model is None:
                error_logger.log_error(
                    "Модель не обучена или не инициализирована",
                    "voice_identification",
                    "predict"
                )
                return []
                
            # Проверка входных данных
            if not features_list or len(features_list) == 0:
                error_logger.log_error(
                    "Пустой список признаков",
                    "voice_identification",
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
                
                # Если уверенность выше порога, идентифицируем пользователя
                if confidence >= 0.8:
                    label = self.classes[predicted_class_index]
                else:
                    label = "Unknown"
                
                # Добавляем результат в список
                results.append({
                    'label': label,
                    'confidence': float(confidence)
                })
            
            return results
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "predict",
                "Ошибка при идентификации пользователя"
            )
            
            return []
