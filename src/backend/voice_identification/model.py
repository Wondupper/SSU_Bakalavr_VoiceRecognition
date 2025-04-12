import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils, regularizers, callbacks
import pickle
import time
from backend.processors.dataset_creator import extract_features
from backend.api.error_logger import error_logger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import lru_cache
import hashlib

# Константы для оптимизации
N_JOBS = max(1, multiprocessing.cpu_count() - 1)  # Оставляем 1 ядро для системы
FEATURE_CACHE_SIZE = 512  # Размер кэша для функции извлечения признаков
USE_THREADING = True      # Использовать многопоточность для предсказаний

# Включение использования GPU с ограничением памяти
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Ограничиваем количество используемой GPU памяти
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        error_logger.log_error(f"Используется GPU: {len(gpus)} устройств", "initialization", "voice_identification")
except Exception as e:
    error_logger.log_error(f"Ошибка при настройке GPU: {str(e)}", "initialization", "voice_identification")

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
        
        # Кэш для весов модели
        self.weights_cache = {}
        
        # Инициализация кэша для признаков
        self.feature_cache = {}
        
        error_logger.log_error(
            "Инициализирована модель идентификации голоса с оптимизациями",
            "initialization",
            "voice_identification"
        )
    
    def _create_model(self):
        """
        Архитектура модели для идентификации голоса
        с улучшенной инициализацией и регуляризацией
        """
        # Размерность для признаков из новой функции extract_features
        inputs = layers.Input(shape=(None, 134))
        
        # Используем более стабильные инициализации весов для лучшей сходимости
        kernel_initializer = 'he_normal'  # Лучше для ReLU активаций
        
        # Первый блок свертки
        x = layers.Conv1D(
            filters=64, 
            kernel_size=3, 
            padding='same', 
            activation='relu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(0.001)  # L2 регуляризация для уменьшения переобучения
        )(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Второй блок свертки
        x = layers.Conv1D(
            filters=128, 
            kernel_size=3, 
            padding='same', 
            activation='relu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Третий блок свертки
        x = layers.Conv1D(
            filters=256, 
            kernel_size=3, 
            padding='same', 
            activation='relu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        
        # Механизм внимания для лучшего фокуса на ключевых частях аудио
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(256)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Применяем внимание
        x = layers.Multiply()([x, attention])
        x = layers.GlobalAveragePooling1D()(x)
        
        # Полносвязные слои с улучшенной регуляризацией
        x = layers.Dense(
            256, 
            activation='relu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(
            128, 
            activation='relu',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        
        # Выходной слой будет динамически настроен в train()
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
    
    def _cached_extract_features(self, audio_data):
        """
        Кэширует извлеченные признаки для избежания повторных вычислений
        
        Args:
            audio_data: аудиоданные для извлечения признаков
            
        Returns:
            Извлеченные признаки
        """
        # Создаем уникальный хэш для аудиоданных
        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
        
        # Проверяем, есть ли признаки в кэше
        if audio_hash in self.feature_cache:
            return self.feature_cache[audio_hash]
        
        # Если нет, извлекаем признаки
        features = extract_features(audio_data)
        
        # Ограничиваем размер кэша
        if len(self.feature_cache) >= FEATURE_CACHE_SIZE:
            # Удаляем случайный элемент кэша для освобождения места
            try:
                keys = list(self.feature_cache.keys())
                if keys:
                    del self.feature_cache[keys[0]]
            except Exception as e:
                error_logger.log_error(f"Ошибка при очистке кэша признаков: {str(e)}", "training", "voice_identification")
        
        # Кэшируем результат
        self.feature_cache[audio_hash] = features
        
        return features

    def train(self, dataset, progress_callback=None):
        """
        Обучение модели с оптимизацией: кэширование признаков, 
        сохранение промежуточных весов, использование TF.data
        """
        if not dataset or len(dataset) == 0:
            raise ValueError("Пустой набор данных")
        
        # Очищаем кэш признаков перед обучением
        self.feature_cache = {}
        
        # Сбрасываем модель, если она уже обучена
        if self.is_trained:
            self._create_model()
        
        # Извлечение признаков из датасета
        all_features = []
        all_labels = []
        unique_labels = []
        
        # Собираем все уникальные метки
        for item in dataset:
            if item['label'] not in unique_labels:
                unique_labels.append(item['label'])
        
        # Преобразуем метки в числовые значения
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Извлекаем признаки и метки
        for item in dataset:
            try:
                # Используем кэширование для извлечения признаков
                features = self._cached_extract_features(item['features'])
                all_features.append(features)
                
                # One-hot encoding для меток
                label_index = label_to_index[item['label']]
                label_onehot = np.zeros(len(unique_labels))
                label_onehot[label_index] = 1
                all_labels.append(label_onehot)
            except Exception as e:
                error_logger.log_error(f"Ошибка при обработке примера: {str(e)}", "training", "voice_identification")
                continue
        
        # Проверка наличия данных после обработки
        if not all_features or len(all_features) == 0:
            raise ValueError("Не удалось извлечь признаки из датасета")
        
        # Проверка и стандартизация размерности признаков
        expected_features = 134
        for i, features in enumerate(all_features):
            if features.shape[1] != expected_features:
                if features.shape[1] < expected_features:
                    # Если признаков меньше, дополняем нулями
                    padded = np.zeros((features.shape[0], expected_features))
                    padded[:, :features.shape[1]] = features
                    all_features[i] = padded
                else:
                    # Если признаков больше, обрезаем
                    all_features[i] = features[:, :expected_features]
        
        # Объединение всех признаков в один массив
        features = np.vstack(all_features)
        y = np.array(all_labels)
        
        # Определяем количество классов
        num_classes = len(unique_labels)
        if num_classes <= 1:
            error_logger.log_error(
                "Только один класс предоставлен для обучения. Дополнительные проверки не выполняются.",
                "training",
                "voice_identification"
            )
        
        # Настройка размера батча в зависимости от размера датасета
        if len(dataset) < 10:
            batch_size = 2
        elif len(dataset) < 30:
            batch_size = 4
        elif len(dataset) < 100:
            batch_size = 8
        else:
            batch_size = 16
        
        # Создаем TensorFlow Dataset для оптимизации загрузки данных
        train_dataset = tf.data.Dataset.from_tensor_slices((features, y))
        train_dataset = train_dataset.shuffle(buffer_size=len(features))
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Оптимизация загрузки данных
        
        # Настройка модели для обучения
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Создание колбэков для обучения
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001
            ),
            callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: error_logger.log_error(
                    f"Эпоха {epoch+1}: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}, lr={float(tf.keras.backend.get_value(self.model.optimizer.lr)):.6f}",
                    "training_progress",
                    "voice_identification"
                ) if (epoch+1) % 5 == 0 else None
            ),
            # Новый колбэк для сохранения промежуточных весов модели
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._cache_model_weights(epoch) if (epoch+1) % 10 == 0 else None
            )
        ]
        
        # Добавляем колбэк прогресса, если он предоставлен
        if progress_callback:
            callbacks_list.append(
                callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: progress_callback.on_epoch_end(epoch, logs)
                )
            )
        
        # Расчет весов классов для балансировки, если классы несбалансированы
        class_weights = {}
        if num_classes > 1:
            # Подсчитываем количество примеров для каждого класса
            class_counts = np.sum(y, axis=0)
            total_samples = np.sum(class_counts)
            
            # Вычисляем веса обратно пропорционально частоте класса
            for i in range(num_classes):
                # Проверка на нулевой счетчик (на всякий случай)
                if class_counts[i] > 0:
                    # Формула: total_samples / (num_classes * class_count)
                    class_weights[i] = total_samples / (num_classes * class_counts[i])
                else:
                    class_weights[i] = 1.0
                    
            error_logger.log_error(
                f"Рассчитаны веса классов для балансировки: {class_weights}",
                "training",
                "voice_identification"
            )
        
        # Обучение модели с увеличенным числом эпох и улучшенными параметрами
        try:
            history = self.model.fit(
                train_dataset,  # Используем TF Dataset вместо массивов
                epochs=100,     # Увеличиваем максимальное число эпох
                callbacks=callbacks_list,
                class_weight=class_weights if num_classes > 1 else None,
                verbose=1
            )
            
            # Логируем результаты обучения
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            best_epoch = np.argmin(history.history['loss']) + 1
            
            error_logger.log_error(
                f"Обучение модели идентификации завершено. Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}, "
                f"Лучшая эпоха: {best_epoch}/{len(history.history['loss'])}",
                "training",
                "voice_identification"
            )
            
        except Exception as e:
            error_msg = f"Ошибка при обучении модели: {str(e)}"
            error_logger.log_error(error_msg, "training", "voice_identification")
            
            # Пытаемся восстановить веса из кэша, если произошла ошибка
            if self.weights_cache:
                last_epoch = max(self.weights_cache.keys())
                error_logger.log_error(
                    f"Восстанавливаем веса из кэша для эпохи {last_epoch}",
                    "training", 
                    "voice_identification"
                )
                self.model.set_weights(self.weights_cache[last_epoch])
            else:
                raise ValueError(error_msg)
        
        # Сохранение меток пользователей
        self.user_labels = unique_labels
        self.is_trained = True
        
        # Очистка кэша после успешного обучения
        self.weights_cache = {}
        
        return history
    
    def _cache_model_weights(self, epoch):
        """
        Кэширует веса модели для возможного восстановления
        
        Args:
            epoch: номер эпохи
        """
        try:
            self.weights_cache[epoch] = self.model.get_weights()
            
            # Ограничиваем размер кэша - храним только последние 3 сохранения
            if len(self.weights_cache) > 3:
                min_epoch = min(self.weights_cache.keys())
                del self.weights_cache[min_epoch]
        except Exception as e:
            error_logger.log_error(f"Ошибка при кэшировании весов модели: {str(e)}", "training", "voice_identification")

    def predict(self, audio_fragments):
        """
        Предсказание имени пользователя по аудиофрагментам с улучшенным
        механизмом принятия решений и параллельной обработкой
        """
        # Определяем порог уверенности на основе количества классов
        # Чем больше классов, тем ниже должен быть порог
        num_classes = len(self.user_labels) if self.user_labels else 0
        
        if num_classes <= 2:
            CONFIDENCE_THRESHOLD = 0.7  # Высокий порог для 1-2 классов
        elif num_classes <= 5:
            CONFIDENCE_THRESHOLD = 0.6  # Средний порог для 3-5 классов
        elif num_classes <= 10:
            CONFIDENCE_THRESHOLD = 0.5  # Умеренный порог для 6-10 классов
        else:
            CONFIDENCE_THRESHOLD = 0.4  # Низкий порог для многих классов
        
        # Проверка, что модель обучена
        if not self.is_trained or not self.user_labels:
            error_logger.log_error(
                "Попытка предсказания с необученной моделью",
                "prediction",
                "voice_identification"
            )
            raise ValueError("Модель не обучена")
        
        # Проверка на наличие фрагментов
        if not audio_fragments or len(audio_fragments) == 0:
            error_logger.log_error(
                "Отсутствуют аудиофрагменты для анализа",
                "prediction",
                "voice_identification"
            )
            raise ValueError("Отсутствуют аудиофрагменты для анализа")
        
        # Используем параллельную обработку для извлечения признаков
        # если фрагментов достаточно и можно использовать многопоточность
        features_list = []
        use_parallel = len(audio_fragments) >= 4 and N_JOBS > 1 and USE_THREADING
        
        if use_parallel:
            try:
                # Используем ThreadPoolExecutor для CPU-bound задач, которые будут работать с TensorFlow
                with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
                    features_list = list(executor.map(extract_features, audio_fragments))
            except Exception as e:
                error_logger.log_error(
                    f"Ошибка при параллельном извлечении признаков: {str(e)}", 
                    "prediction", 
                    "voice_identification"
                )
                # Если параллельная обработка не удалась, переходим к последовательной
                features_list = []
                for fragment in audio_fragments:
                    try:
                        features = extract_features(fragment)
                        features_list.append(features)
                    except Exception as e:
                        error_logger.log_error(
                            f"Ошибка при извлечении признаков: {str(e)}",
                            "prediction", 
                            "voice_identification"
                        )
                        continue
        else:
            # Последовательное извлечение признаков
            for fragment in audio_fragments:
                try:
                    features = extract_features(fragment)
                    features_list.append(features)
                except Exception as e:
                    error_logger.log_error(
                        f"Ошибка при извлечении признаков: {str(e)}",
                        "prediction", 
                        "voice_identification"
                    )
                    continue
        
        # Проверка на наличие признаков после извлечения
        if not features_list or len(features_list) == 0:
            error_logger.log_error(
                "Не удалось извлечь признаки из аудиофрагментов",
                "prediction",
                "voice_identification"
            )
            raise ValueError("Не удалось извлечь признаки из аудиофрагментов")
        
        # Анализ каждого фрагмента отдельно для голосования
        predictions_per_fragment = []
        confidences_per_fragment = []
        
        # Готовим данные для предсказания
        processed_features = []
        expected_features = 134  # Должно соответствовать извлекаемым признакам
        
        for features in features_list:
            try:
                # Обеспечиваем правильную форму для входа в модель
                if len(features.shape) != 2:
                    error_logger.log_error(
                        f"Неверная размерность признаков: {features.shape}",
                        "prediction",
                        "voice_identification"
                    )
                    continue
                
                # Проверка и исправление размерности признаков
                if features.shape[1] != expected_features:
                    if features.shape[1] < expected_features:
                        # Если признаков меньше, дополняем нулями
                        padded = np.zeros((features.shape[0], expected_features))
                        padded[:, :features.shape[1]] = features
                        features = padded
                    else:
                        # Если признаков больше, обрезаем
                        features = features[:, :expected_features]
                
                # Преобразование в формат для модели
                input_data = np.expand_dims(features, axis=0)
                processed_features.append(input_data)
                
            except Exception as e:
                error_logger.log_error(
                    f"Ошибка при подготовке данных для предсказания: {str(e)}",
                    "prediction",
                    "voice_identification"
                )
                continue
        
        # Выполняем предсказания для всех фрагментов
        for i, input_data in enumerate(processed_features):
            try:
                # Предсказание и получение вероятностей
                predictions = self.model.predict(input_data, verbose=0)
                
                # Находим класс с максимальной вероятностью
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                
                # Добавляем результат голосования
                predictions_per_fragment.append(predicted_class)
                confidences_per_fragment.append(confidence)
                
            except Exception as e:
                error_logger.log_error(
                    f"Ошибка при предсказании для фрагмента {i}: {str(e)}",
                    "prediction",
                    "voice_identification"
                )
                continue
        
        # Если ни одно предсказание не получено
        if not predictions_per_fragment:
            return "unknown"
        
        # Анализируем результаты голосования с улучшенным механизмом
        # Используем взвешенное голосование с учетом уверенности
        vote_counts = {}
        
        for pred_class, confidence in zip(predictions_per_fragment, confidences_per_fragment):
            if pred_class not in vote_counts:
                vote_counts[pred_class] = 0
            
            # Голос взвешивается уверенностью
            vote_counts[pred_class] += confidence
        
        # Находим класс с наибольшим взвешенным голосованием
        if vote_counts:
            best_class = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            best_confidence = vote_counts[best_class] / len(predictions_per_fragment)
            
            # Проверка индекса класса
            if best_class < 0 or best_class >= len(self.user_labels):
                error_logger.log_error(
                    f"Недопустимый индекс класса: {best_class}, количество классов: {len(self.user_labels)}",
                    "prediction",
                    "voice_identification"
                )
                return "unknown"
            
            # Проверка уверенности
            if best_confidence < CONFIDENCE_THRESHOLD:
                error_logger.log_error(
                    f"Недостаточная уверенность: {best_confidence:.4f} < {CONFIDENCE_THRESHOLD}",
                    "prediction",
                    "voice_identification"
                )
                return "unknown"
            
            # Возвращаем имя пользователя
            return self.user_labels[best_class]
        
        return "unknown"
    
    def reset(self):
        """Сброс модели"""
        self._create_model()
        self.user_labels = []
        self.is_trained = False
        return True
    
    def save(self):
        """Сохранение модели на диск"""
        if not self.is_trained:
            return None
        
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'voice_id_model_{timestamp}')
        
        try:
            # Сохраняем веса модели в бинарный файл
            self.model.save_weights(model_path + '.h5')
            
            # Сохраняем метаданные в отдельный файл
            metadata = {
                'user_labels': self.user_labels,
                'is_trained': self.is_trained,
                'timestamp': timestamp
            }
            
            with open(model_path + '_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
                
            return model_path
        except Exception as e:
            error_logger.log_error(f"Ошибка при сохранении модели: {str(e)}", "model", "voice_identification")
            return None
    
    def load(self, path):
        """Загрузка модели с диска"""
        try:
            # Загрузка метаданных
            with open(path + '_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            # Восстановление состояния объекта
            self.user_labels = metadata['user_labels']
            self.is_trained = metadata['is_trained']
            
            # Перестройка модели для правильного количества классов
            if self.is_trained and self.user_labels:
                num_classes = len(self.user_labels)
                if num_classes > 1:
                    # Пересоздаем модель с правильным количеством выходов
                    self._create_model()
                    
                    # Настраиваем выходной слой
                    model_config = self.model.get_config()
                    output_layer_config = model_config['layers'][-1]['config']
                    output_layer_config['units'] = num_classes
                    output_layer_config['activation'] = 'softmax'
                    
                    # Создаем новую модель с обновленной конфигурацией
                    new_model = models.Model.from_config(model_config)
                    self.model = new_model
                    
                    # Компилируем модель
                    self.model.compile(
                        optimizer=optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
            
            # Загрузка весов модели
            self.model.load_weights(path + '.h5')
            
            return True
        except Exception as e:
            error_logger.log_error(f"Ошибка при загрузке модели: {str(e)}", "model", "voice_identification")
            return False
