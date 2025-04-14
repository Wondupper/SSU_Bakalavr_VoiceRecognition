import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import pickle
import time
from backend.processors.dataset_creator import extract_features
from backend.api.error_logger import error_logger
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import hashlib
import sys

# Константы для оптимизации
N_JOBS = max(1, multiprocessing.cpu_count() - 1)  # Оставляем 1 ядро для системы
FEATURE_CACHE_SIZE = 512  # Размер кэша для функции извлечения признаков
USE_THREADING = True      # Использовать многопоточность для предсказаний

# Настройка GPU для TensorFlow (применяем только если не настроено в другом месте)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Ограничиваем количество используемой GPU памяти
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        error_logger.log_error(f"Используется GPU для эмоций: {len(gpus)} устройств", "initialization", "emotion_recognition")
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
    line_no = exc_tb.tb_lineno
    print(f"{fname} - {line_no} - {str(e)}")
    error_logger.log_error(f"Ошибка при настройке GPU для эмоций: {str(e)}", "initialization", "emotion_recognition")

class EmotionRecognitionModel:
    def __init__(self):
        self.model = None
        self.emotion_labels = ['гнев', 'радость', 'грусть']
        self.is_trained = False
        # Исправляем путь к директории для сохранения
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'emotion_recognition')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
        self._create_model()
        
        # Инициализация кэша для весов и признаков
        self.weights_cache = {}
        self.feature_cache = {}
        
        error_logger.log_error(
            "Инициализирована модель распознавания эмоций с оптимизациями",
            "initialization",
            "emotion_recognition"
        )
    
    def _create_model(self):
        """
        Создание улучшенной модели для распознавания эмоций с более глубокой архитектурой
        и механизмом внимания
        """
        # Размерность входных данных
        input_shape = (None, 134)
        inputs = layers.Input(shape=input_shape)
        
        # Применяем слой маскирования для поддержки изменяющейся длины входных последовательностей
        masking = layers.Masking(mask_value=0.0)(inputs)
        
        # 1. Сверточные блоки с разными размерами ядра для захвата признаков разных масштабов
        # Первый путь: мелкие детали
        conv1_1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(masking)
        bn1_1 = layers.BatchNormalization()(conv1_1)
        pool1_1 = layers.MaxPooling1D(pool_size=2)(bn1_1)
        dropout1_1 = layers.SpatialDropout1D(0.2)(pool1_1)
        
        # Второй путь: средние детали
        conv1_2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(masking)
        bn1_2 = layers.BatchNormalization()(conv1_2)
        pool1_2 = layers.MaxPooling1D(pool_size=2)(bn1_2)
        dropout1_2 = layers.SpatialDropout1D(0.2)(pool1_2)
        
        # Третий путь: крупные детали
        conv1_3 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(masking)
        bn1_3 = layers.BatchNormalization()(conv1_3)
        pool1_3 = layers.MaxPooling1D(pool_size=2)(bn1_3)
        dropout1_3 = layers.SpatialDropout1D(0.2)(pool1_3)
        
        # Объединение результатов разных путей
        concat1 = layers.Concatenate()([dropout1_1, dropout1_2, dropout1_3])
        
        # 2. Дилатированные свертки для увеличения рецептивного поля
        conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', 
                              dilation_rate=2,
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(concat1)
        bn2 = layers.BatchNormalization()(conv2)
        dropout2 = layers.SpatialDropout1D(0.2)(bn2)
        
        conv3 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', 
                              dilation_rate=4,
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout2)
        bn3 = layers.BatchNormalization()(conv3)
        dropout3 = layers.SpatialDropout1D(0.3)(bn3)
        
        # 3. Механизм внимания для фокусировки на значимых частях аудио
        # Слой внимания
        attention = layers.Dense(1, activation='tanh')(dropout3)
        attention = layers.Flatten()(attention)
        attention_weights = layers.Activation('softmax')(attention)
        
        # Применяем веса внимания
        attention_weights = layers.RepeatVector(128)(attention_weights)
        attention_weights = layers.Permute([2, 1])(attention_weights)
        attention_output = layers.Multiply()([dropout3, attention_weights])
        
        # 4. Глобальные пулинги (используем разные типы пулинга для разных аспектов эмоций)
        avg_pool = layers.GlobalAveragePooling1D()(attention_output)
        max_pool = layers.GlobalMaxPooling1D()(attention_output)
        
        # Объединяем результаты пулингов
        concat2 = layers.Concatenate()([avg_pool, max_pool])
        
        # 5. Полносвязные слои с регуляризацией
        fc1 = layers.Dense(256, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))(concat2)
        bn4 = layers.BatchNormalization()(fc1)
        dropout4 = layers.Dropout(0.4)(bn4)
        
        fc2 = layers.Dense(128, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001))(dropout4)
        bn5 = layers.BatchNormalization()(fc2)
        dropout5 = layers.Dropout(0.3)(bn5)
        
        # 6. Выходной слой с softmax активацией
        outputs = layers.Dense(len(self.emotion_labels), activation='softmax')(dropout5)
        
        # Инициализация модели
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Компиляция модели с улучшенными параметрами оптимизатора
        self.model.compile(
            optimizer=optimizers.Adam(
                learning_rate=0.0003,  # Снижаем скорость обучения
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                clipnorm=1.0  # Ограничиваем градиенты для стабильности
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
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
        
        # Если нет, извлекаем признаки для эмоций (с флагом for_emotion=True)
        features = extract_features(audio_data, for_emotion=True)
        
        # Ограничиваем размер кэша
        if len(self.feature_cache) >= FEATURE_CACHE_SIZE:
            # Удаляем случайный элемент кэша для освобождения места
            try:
                keys = list(self.feature_cache.keys())
                if keys:
                    del self.feature_cache[keys[0]]
            except Exception as e:
                error_logger.log_error(f"Ошибка при очистке кэша признаков эмоций: {str(e)}", "training", "emotion_recognition")
        
        # Кэшируем результат
        self.feature_cache[audio_hash] = features
        
        return features

    def train(self, dataset, progress_callback=None):
        """
        Обучение модели с оптимизацией: кэширование признаков, 
        сохранение промежуточных весов, TF.data и параллельная обработка
        
        Args:
            dataset: Список словарей с ключами 'features' и 'label'
            progress_callback: Функция обратного вызова для отслеживания прогресса
            
        Returns:
            История обучения
        """
        if not dataset or len(dataset) == 0:
            raise ValueError("Пустой набор данных для обучения")
        
        # Очищаем кэш признаков перед обучением
        self.feature_cache = {}
        
        # Сбрасываем модель, если она уже обучена
        if self.is_trained:
            self._create_model()
        
        # Извлечение признаков и меток из датасета
        all_features = []
        all_labels = []
        
        # Собираем все уникальные эмоции
        unique_emotions = set()
        for item in dataset:
            unique_emotions.add(item['label'])
        
        self.emotion_labels = sorted(list(unique_emotions))
        
        error_logger.log_error(
            f"Обнаружены эмоции для обучения: {self.emotion_labels}",
            "training",
            "emotion_recognition"
        )
        
        # Преобразуем метки в числовые значения
        label_to_index = {label: idx for idx, label in enumerate(self.emotion_labels)}
        
        # Параллельное извлечение признаков, если датасет достаточно большой
        if len(dataset) >= 10 and N_JOBS > 1:
            try:
                # Подготавливаем задачи для параллельной обработки
                extraction_tasks = []
                for item in dataset:
                    extraction_tasks.append((item['features'], item['label']))
                
                # Функция обработки одного элемента
                def process_item(task):
                    audio_data, label = task
                    try:
                        features = self._cached_extract_features(audio_data)
                        label_index = label_to_index[label]
                        return (features, label_index)
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                        line_no = exc_tb.tb_lineno
                        print(f"{fname} - {line_no} - {str(e)}")
                        error_logger.log_error(f"Ошибка при обработке примера: {str(e)}", "training", "emotion_recognition")
                        return None
                
                # Параллельное выполнение
                with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
                    results = list(executor.map(process_item, extraction_tasks))
                
                # Обработка результатов
                for result in results:
                    if result is not None:
                        features, label_index = result
                        all_features.append(features)
                        
                        # One-hot encoding для меток
                        label_onehot = np.zeros(len(self.emotion_labels))
                        label_onehot[label_index] = 1
                        all_labels.append(label_onehot)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                error_logger.log_error(f"Ошибка при параллельной обработке: {str(e)}", "training", "emotion_recognition")
                # Возвращаемся к последовательной обработке
                for item in dataset:
                    try:
                        features = self._cached_extract_features(item['features'])
                        all_features.append(features)
                        
                        # One-hot encoding для меток
                        label_index = label_to_index[item['label']]
                        label_onehot = np.zeros(len(self.emotion_labels))
                        label_onehot[label_index] = 1
                        all_labels.append(label_onehot)
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                        line_no = exc_tb.tb_lineno
                        print(f"{fname} - {line_no} - {str(e)}")
                        error_logger.log_error(f"Ошибка при обработке примера: {str(e)}", "training", "emotion_recognition")
                        continue
        else:
            # Последовательная обработка для небольших датасетов
            for item in dataset:
                try:
                    features = self._cached_extract_features(item['features'])
                    all_features.append(features)
                    
                    # One-hot encoding для меток
                    label_index = label_to_index[item['label']]
                    label_onehot = np.zeros(len(self.emotion_labels))
                    label_onehot[label_index] = 1
                    all_labels.append(label_onehot)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                    line_no = exc_tb.tb_lineno
                    print(f"{fname} - {line_no} - {str(e)}")
                    error_logger.log_error(f"Ошибка при обработке примера: {str(e)}", "training", "emotion_recognition")
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
        
        # Создаем TensorFlow Dataset для оптимизации загрузки данных
        train_dataset = tf.data.Dataset.from_tensor_slices((features, y))
        train_dataset = train_dataset.shuffle(buffer_size=len(features))
        
        # Настройка размера батча в зависимости от размера датасета
        if len(dataset) < 10:
            batch_size = 2
        elif len(dataset) < 30:
            batch_size = 4
        elif len(dataset) < 100:
            batch_size = 8
        else:
            batch_size = 16
        
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Оптимизация загрузки данных
        
        # Настраиваем оптимизатор с более оптимальными параметрами
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Создаем колбэки для обучения с улучшенными параметрами
        model_callbacks = [
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
                    "emotion_recognition"
                ) if (epoch+1) % 5 == 0 else None
            ),
            # Новый колбэк для сохранения промежуточных весов модели
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._cache_model_weights(epoch) if (epoch+1) % 10 == 0 else None
            )
        ]
        
        # Добавляем колбэк прогресса в список колбэков, если он предоставлен
        if progress_callback:
            model_callbacks.append(
                callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: progress_callback.on_epoch_end(epoch, logs)
                )
            )
        
        # Расчет весов классов для балансировки, если классы несбалансированы
        class_weights = {}
        num_classes = len(self.emotion_labels)
        
        if num_classes > 1:
            # Подсчитываем количество примеров для каждого класса
            class_counts = np.sum(y, axis=0)
            total_samples = len(y)
            
            # Вычисляем веса обратно пропорционально частоте класса
            for i in range(num_classes):
                if class_counts[i] > 0:
                    class_weights[i] = total_samples / (num_classes * class_counts[i])
                else:
                    class_weights[i] = 1.0
                
            error_logger.log_error(
                f"Рассчитаны веса классов для балансировки эмоций: {class_weights}",
                "training",
                "emotion_recognition"
            )
        
        # Обучение модели с улучшенными параметрами
        try:
            history = self.model.fit(
                train_dataset,  # Используем TF Dataset вместо массивов
                epochs=100,  # То же количество эпох
                callbacks=model_callbacks,
                class_weight=class_weights if num_classes > 1 else None,
                verbose=1
            )
            
            # Логирование результатов обучения
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            best_epoch = np.argmin(history.history['loss']) + 1
            
            error_logger.log_error(
                f"Обучение модели эмоций завершено. Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}, "
                f"Лучшая эпоха: {best_epoch}/{len(history.history['loss'])}",
                "training",
                "emotion_recognition"
            )
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            error_logger.log_error(f"Ошибка при обучении модели эмоций: {str(e)}", "training", "emotion_recognition")
            
            # Пытаемся восстановить веса из кэша, если произошла ошибка
            if self.weights_cache:
                last_epoch = max(self.weights_cache.keys())
                error_logger.log_error(
                    f"Восстанавливаем веса из кэша для эпохи {last_epoch}",
                    "training", 
                    "emotion_recognition"
                )
                self.model.set_weights(self.weights_cache[last_epoch])
            else:
                raise ValueError("Не удалось восстановить веса из кэша")
        
        self.is_trained = True
        
        # Очистка кэша после успешного обучения
        self.weights_cache = {}
        
        # Возвращаем историю обучения для дальнейшего анализа, если необходимо
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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            error_logger.log_error(f"Ошибка при кэшировании весов модели эмоций: {str(e)}", "training", "emotion_recognition")

    def predict(self, audio_fragments):
        """
        Предсказание эмоции по аудиофрагментам с параллельной обработкой
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
        
        # Используем параллельную обработку для извлечения признаков
        # если фрагментов достаточно и можно использовать многопоточность
        features_list = []
        use_parallel = len(audio_fragments) >= 4 and N_JOBS > 1 and USE_THREADING
        
        # Для распознавания эмоций используем дополнительный флаг for_emotion=True
        extract_func = lambda audio: extract_features(audio, for_emotion=True)
        
        if use_parallel:
            try:
                # Используем ThreadPoolExecutor для извлечения признаков
                with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
                    features_list = list(executor.map(extract_func, audio_fragments))
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                error_logger.log_error(
                    f"Ошибка при параллельном извлечении признаков эмоций: {str(e)}", 
                    "prediction", 
                    "emotion_recognition"
                )
                # Если параллельная обработка не удалась, переходим к последовательной
                features_list = []
                for fragment in audio_fragments:
                    try:
                        features = extract_func(fragment)
                        features_list.append(features)
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                        line_no = exc_tb.tb_lineno
                        print(f"{fname} - {line_no} - {str(e)}")
                        error_logger.log_error(
                            f"Ошибка при извлечении признаков эмоций: {str(e)}",
                            "prediction", 
                            "emotion_recognition"
                        )
                        continue
        else:
            # Последовательное извлечение признаков
            for fragment in audio_fragments:
                try:
                    features = extract_func(fragment)
                    features_list.append(features)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                    line_no = exc_tb.tb_lineno
                    print(f"{fname} - {line_no} - {str(e)}")
                    error_logger.log_error(
                        f"Ошибка при извлечении признаков эмоций: {str(e)}",
                        "prediction", 
                        "emotion_recognition"
                    )
                    continue
        
        # Проверка на наличие признаков после извлечения
        if not features_list or len(features_list) == 0:
            raise ValueError("Не удалось извлечь признаки из аудиофрагментов")
        
        # Среднее значение признаков по всем фрагментам для определения эмоции
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
        confidence = predictions[0][predicted_class]
        
        # Логируем уровень уверенности
        error_logger.log_error(
            f"Предсказана эмоция с уверенностью {confidence:.4f}",
            "prediction",
            "emotion_recognition"
        )
        
        # Проверка индекса класса
        if predicted_class < 0 or predicted_class >= len(self.emotion_labels):
            # Если индекс за пределами допустимых значений, возвращаем дефолтную эмоцию
            return self.emotion_labels[0]  # Возвращаем первую эмоцию как дефолтную
        
        # Возвращение названия эмоции
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
        try:
            # Убедимся, что директория существует
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
            
            timestamp = int(time.time())
            model_path = os.path.join(self.model_dir, f'emotion_model_{timestamp}')
            
            # Сохранение модели TensorFlow с обработкой ошибок
            try:
                self.model.save(f'{model_path}.h5')
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                line_no = exc_tb.tb_lineno
                print(f"{fname} - {line_no} - {str(e)}")
                error_logger.log_error(f"Ошибка при сохранении модели в H5: {str(e)}", "model", "emotion_recognition")
                # Альтернативный метод сохранения, если обычный не сработал
                self.model.save_weights(f'{model_path}_weights.h5')
                print(f"Сохранены только веса модели эмоций: {model_path}_weights.h5")
            
            # Сохранение дополнительных данных
            with open(f'{model_path}_metadata.pkl', 'wb') as f:
                pickle.dump({
                    'is_trained': self.is_trained,
                    'emotion_labels': self.emotion_labels
                }, f)
            
            print(f"Модель распознавания эмоций сохранена в {model_path}")
            return model_path
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            error_logger.log_error(f"Критическая ошибка при сохранении модели: {str(e)}", "model", "emotion_recognition")
            print(f"Не удалось сохранить модель распознавания эмоций: {str(e)}")
            return None
    
    def load(self, path):
        """
        Загрузка модели из файла
        """
        # Загрузка модели TensorFlow
        try:
            self.model = models.load_model(f'{path}.h5')
            
            # Загрузка дополнительных данных
            with open(f'{path}_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
                self.is_trained = metadata['is_trained']
                # Загружаем метки эмоций, если они есть
                if 'emotion_labels' in metadata:
                    self.emotion_labels = metadata['emotion_labels']
            
            return True
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            error_logger.log_error(f"Ошибка при загрузке модели: {str(e)}", "model", "emotion_recognition")
            return False
