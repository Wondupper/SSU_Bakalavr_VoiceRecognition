import os
import numpy as np
import tensorflow as tf
from backend.api.error_logger import error_logger
import sys
from backend.processors.dataset_creators.dataset_creator import extract_features


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
            
            # Полносвязный слой для классификации
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            # Выходной слой с softmax-активацией для многоклассовой классификации
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            # Создаем модель
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            
            # Компилируем модель
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            print(f"{sys.exc_info()[2].tb_frame.f_code.co_filename} - {sys.exc_info()[2].tb_lineno} - {str(e)}")
            return None
                
    def train(self, audio_fragments, labels):
        """
        Обучает модель на наборе аудиофрагментов и соответствующих меток.
        
        Args:
            audio_fragments: Список аудиофрагментов для обучения
            labels: Список меток (имена пользователей) для каждого фрагмента
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        try:
            # Проверка входных данных
            if not audio_fragments or not labels:
                return False
                
            if len(audio_fragments) != len(labels):
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
                
                # Извлекаем признаки из всех аудиофрагментов
                features_list = []
                for fragment in audio_fragments:
                    features = extract_features(fragment)
                    if features is not None:
                        features_list.append(features)
                    else:
                        # В случае ошибки извлечения признаков, пропускаем этот фрагмент
                        return False
                        
                # Проверяем, есть ли извлеченные признаки
                if not features_list:
                    return False
                    
                # Преобразуем список в numpy массив
                X = np.array(features_list)
                
                # Получаем размерность входных данных для модели
                input_shape = (X.shape[1], X.shape[2])
                
                # Создаем новую модель
                self.model = self.build_model(input_shape, len(self.classes))
                
                if self.model is None:
                    return False
                    
                # Обучаем модель с нуля
                history = self.model.fit(
                    X, numeric_labels,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=5,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Устанавливаем флаг, что модель обучена
                self.is_trained = True
                
                # Логируем результаты обучения
                final_loss = history.history['loss'][-1]
                final_accuracy = history.history['accuracy'][-1]
                
                return True
            
            else:
                # Модель уже существует, выполняем дообучение
                
                # Создаем отображение имен на индексы
                label_to_index = {label: i for i, label in enumerate(self.classes)}
                
                # Преобразуем текстовые метки в числовые индексы
                numeric_labels = np.array([label_to_index[label] for label in labels])
                
                # Извлекаем признаки из всех аудиофрагментов
                features_list = []
                for fragment in audio_fragments:
                    features = extract_features(fragment)
                    if features is not None:
                        features_list.append(features)
                        
                # Проверяем, есть ли извлеченные признаки
                if not features_list:
                    return False
                    
                # Преобразуем список в numpy массив
                X = np.array(features_list)
                
                # Дообучаем существующую модель
                history = self.model.fit(
                    X, numeric_labels,
                    epochs=20,  # Меньше эпох для дообучения
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=3,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Устанавливаем флаг, что модель обучена
                self.is_trained = True
                
                # Логируем результаты дообучения
                final_loss = history.history['loss'][-1]
                final_accuracy = history.history['accuracy'][-1]
                
                return True
            
            # Сбрасываем флаг обучения
            self.is_training = False
            return True
            
        except Exception as e:
            print(f"{sys.exc_info()[2].tb_frame.f_code.co_filename} - {sys.exc_info()[2].tb_lineno} - {str(e)}")
            
            # Сбрасываем флаг обучения в случае ошибки
            self.is_training = False
            return False
            
    def identify(self, audio_fragments):
        """
        Идентифицирует пользователя по голосу из аудиофрагментов.
        
        Args:
            audio_fragments: Список аудиофрагментов для идентификации
            
        Returns:
            str: Имя пользователя или "unknown", если не удалось идентифицировать
        """
        try:
            # Проверка состояния модели
            if not self.is_trained or self.model is None:
                return "unknown"
                
            # Проверка входных данных
            if not audio_fragments or len(audio_fragments) == 0:
                return "unknown"
                
            # Извлекаем признаки из всех фрагментов
            features_list = []
            for fragment in audio_fragments:
                features = extract_features(fragment)
                if features is not None:
                    features_list.append(features)
                    
            # Проверяем, удалось ли извлечь признаки
            if not features_list:
                return "unknown"
                
            # Преобразуем список в numpy массив
            X = np.array(features_list)
            
            # Получаем предсказания модели для всех фрагментов
            predictions = self.model.predict(X)
            
            # Усредняем предсказания по всем фрагментам
            avg_prediction = np.mean(predictions, axis=0)
            
            # Находим класс с наибольшей вероятностью
            predicted_class_index = np.argmax(avg_prediction)
            max_confidence = avg_prediction[predicted_class_index]
            
            # Проверяем порог уверенности
            if max_confidence < 0.5:  # Если уверенность ниже порога, считаем пользователя неизвестным
                return "unknown"
                
            # Получаем имя пользователя по индексу класса
            identified_user = self.classes[predicted_class_index]
            
            return identified_user
            
        except Exception as e:
            print(f"{sys.exc_info()[2].tb_frame.f_code.co_filename} - {sys.exc_info()[2].tb_lineno} - {str(e)}")
            
            return "unknown"
            
    def save_model(self, filepath):
        """
        Сохраняет модель в файл.
        
        Args:
            filepath: Путь к файлу для сохранения модели
            
        Returns:
            bool: Успешно ли сохранена модель
        """
        try:
            # Проверка состояния модели
            if not self.is_trained or self.model is None:
                return False
                
            # Создаем директорию для модели, если она не существует
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Сохраняем модель
            self.model.save(filepath)
            
            # Сохраняем список классов в отдельный файл
            classes_file = os.path.join(os.path.dirname(filepath), "classes.txt")
            with open(classes_file, 'w') as f:
                for cls in self.classes:
                    f.write(f"{cls}\n")
                    
            return True
            
        except Exception as e:
            print(f"{sys.exc_info()[2].tb_frame.f_code.co_filename} - {sys.exc_info()[2].tb_lineno} - {str(e)}")
            
            return False
            
    def load_model(self, filepath):
        """
        Загружает модель из файла.
        
        Args:
            filepath: Путь к файлу с сохраненной моделью
            
        Returns:
            bool: Успешно ли загружена модель
        """
        try:
            # Проверяем существование файла модели
            if not os.path.exists(filepath):
                return False
                
            # Проверяем существование файла с классами
            classes_file = os.path.join(os.path.dirname(filepath), "classes.txt")
            if not os.path.exists(classes_file):
                return False
                
            # Загружаем модель
            self.model = tf.keras.models.load_model(filepath)
            
            # Загружаем список классов
            self.classes = []
            with open(classes_file, 'r') as f:
                for line in f:
                    self.classes.append(line.strip())
                    
            # Устанавливаем флаг, что модель обучена
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"{sys.exc_info()[2].tb_frame.f_code.co_filename} - {sys.exc_info()[2].tb_lineno} - {str(e)}")
            
            # Сбрасываем модель в случае ошибки
            self.model = None
            self.classes = []
            self.is_trained = False
            
            return False
