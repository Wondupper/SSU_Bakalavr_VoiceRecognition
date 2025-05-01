import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import io
import random
from typing import List, Dict, Tuple, Union, Optional, Any, Set, TypeVar, Generic, cast
from werkzeug.datastructures import FileStorage
from backend.api.error_logger import error_logger
from backend.api.info_logger import info_logger
from backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH, AUGMENTATION, MODELS_PARAMS

T = TypeVar('T', bound=nn.Module)

class AudioModelBase(Generic[T]):
    """
    Базовый класс для моделей обработки аудио.
    
    Атрибуты:
        model: Модель PyTorch 
        classes: Словарь классов и соответствующих им индексов
        index_to_class: Словарь индексов и соответствующих им классов
        device: Устройство для обучения модели (CPU/GPU)
        module_name: Имя модуля для логирования
    """
    
    def __init__(self, module_name: str) -> None:
        """
        Инициализирует базовую модель для обработки аудио.
        
        Args:
            module_name: Имя модуля для логирования
        """
        info_logger.info(f"---Start initializing {module_name} model---")
        # Инициализируем атрибуты
        self.model: Optional[T] = None
        self.classes: Dict[str, int] = {}  # Словарь {класс: индекс}
        self.index_to_class: Dict[int, str] = {}  # Словарь {индекс: класс}
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module_name = module_name
        info_logger.info(f"---Finish initializing {module_name} model---")
        
    @property
    def is_trained(self) -> bool:
        """
        Проверяет, обучена ли модель
        
        Returns:
            bool: True, если модель обучена, иначе False
        """
        return self.model is not None and len(self.classes) > 0
        
    def _apply_augmentation(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Применяет аугментацию к аудиофайлу для расширения обучающей выборки.
        
        Args:
            waveform: Тензор аудио [channels, time]
            
        Returns:
            Список аугментированных аудиофайлов
        """
        info_logger.info(f"---Start augmentation process in {self.module_name} model---")
        try:
            augmented_waveforms: List[torch.Tensor] = [waveform]  # Добавляем оригинальное аудио
            
            new_augmented_waveforms: List[torch.Tensor] = []
            info_logger.info(f"Start augmentation process (FAST_SPEED) in {self.module_name} model")
            # 1. Ускорение аудио (time stretching)
            for speed in AUGMENTATION['FAST_SPEEDS']:
                effects: List[List[str]] = [
                    ["speed", str(speed)],
                    ["rate", str(SAMPLE_RATE)]
                ]
                aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, SAMPLE_RATE, effects)
                new_augmented_waveforms.append(aug_waveform)
            info_logger.info(f"End augmentation process (FAST_SPEED) in {self.module_name} model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info(f"Start augmentation process (SLOW_SPEED) in {self.module_name} model")
            # 2. Замедление аудио
            for speed in AUGMENTATION['SLOW_SPEEDS']:
                effects = [
                    ["speed", str(speed)],
                    ["rate", str(SAMPLE_RATE)]
                ]
                aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, SAMPLE_RATE, effects)
                new_augmented_waveforms.append(aug_waveform)
            info_logger.info(f"End augmentation process (SLOW_SPEED) in {self.module_name} model")

            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info(f"Start augmentation process (REVERBIRATION) in {self.module_name} model")
            # 3. Реверберация (добавление эхо)
            for decay in AUGMENTATION['DECAYS']:
                reverb_waveform: torch.Tensor = waveform.clone()
                # Создаем простую реверберацию, добавляя задержанную и затухающую копию сигнала
                delay_samples: int = int(0.05 * SAMPLE_RATE)  # 50 мс задержка
                if waveform.size(1) > delay_samples:
                    reverb: torch.Tensor = torch.zeros_like(waveform)
                    reverb[:, delay_samples:] = waveform[:, :-delay_samples] * decay
                    reverb_waveform = waveform + reverb
                    # Нормализация
                    reverb_waveform = reverb_waveform / (torch.max(torch.abs(reverb_waveform)) + 1e-6)
                    new_augmented_waveforms.append(reverb_waveform)
            info_logger.info(f"End augmentation process (REVERBIRATION) in {self.module_name} model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info(f"Start augmentation process (TIME_MASKING) in {self.module_name} model")
            # 4. Маскирование по времени (Time Masking)
            for mask_param in AUGMENTATION['MASK_PARAMS']:
                mask_waveform: torch.Tensor = waveform.clone()
                time_mask_samples: int = int(mask_param * waveform.size(1))
                if time_mask_samples > 0:
                    mask_start: int = random.randint(0, waveform.size(1) - time_mask_samples)
                    mask_waveform[:, mask_start:mask_start + time_mask_samples] = 0
                    new_augmented_waveforms.append(mask_waveform)
            info_logger.info(f"End augmentation process (TIME_MASKING) in {self.module_name} model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info(f"Start augmentation process (EHO_ADDING) in {self.module_name} model")
            # 5. Добавление шума
            for snr_db in AUGMENTATION['SNR_DBS']:
                noise: torch.Tensor = torch.randn_like(waveform)
                # Рассчитываем энергию сигнала и шума
                signal_power: torch.Tensor = torch.mean(waveform ** 2)
                noise_power: torch.Tensor = torch.mean(noise ** 2)
                # Корректируем шум для достижения нужного SNR
                snr: float = 10 ** (snr_db / 10)
                noise_scale: torch.Tensor = torch.sqrt(signal_power / (noise_power * snr))
                scaled_noise: torch.Tensor = noise * noise_scale
                # Добавляем шум к сигналу
                noisy_waveform: torch.Tensor = waveform + scaled_noise
                # Нормализация
                noisy_waveform = noisy_waveform / (torch.max(torch.abs(noisy_waveform)) + 1e-6)
                new_augmented_waveforms.append(noisy_waveform)
            info_logger.info(f"End augmentation process (EHO_ADDING) in {self.module_name} model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
                
            return augmented_waveforms
            
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "_apply_augmentation",
                "Ошибка при аугментации аудио"
            )
            # В случае ошибки возвращаем только оригинальное аудио
            return [waveform]
            
        finally:
            info_logger.info(f"---End augmentation process in {self.module_name} model---")
    
    def _extract_features(self, audio_file: FileStorage) -> List[torch.Tensor]:
        """
        Извлекает признаки из аудиофайла с помощью torchaudio
        
        Args:
            audio_file: Файл аудио (объект FileStorage Flask)
            
        Returns:
            Список тензоров признаков для каждого фрагмента
        """
        info_logger.info(f"---Start extract features process in {self.module_name} model---")
        try:
            # Попытка загрузить аудиофайл напрямую из памяти
            temp_filename: Optional[str] = None
            try:
                info_logger.info("Try to save audiofile in buffer")
                # Сохраняем содержимое файла в буфер
                audio_buffer: io.BytesIO = io.BytesIO(audio_file.read())
                # Сбрасываем указатель в начало буфера
                audio_buffer.seek(0)
                # Пытаемся загрузить аудио из буфера
                waveform: torch.Tensor
                sample_rate: int
                waveform, sample_rate = torchaudio.load(audio_buffer)
                # Сбрасываем указатель файла на начало для возможного дальнейшего использования
                audio_file.seek(0)
                info_logger.info("Success")
            except Exception as load_error:
                info_logger.info("Cant save audiofile in buffer, save as temp file")
                # Если не удалось загрузить напрямую, используем временный файл
                temp_id: int = random.randint(1000, 9999)
                temp_filename = f"temp_audio_{self.module_name}_{temp_id}.wav"
                info_logger.info(f"save as {temp_filename}")
                audio_file.save(temp_filename)
                waveform, sample_rate = torchaudio.load(temp_filename)
                # Сбрасываем указатель файла на начало
                audio_file.seek(0)
                info_logger.info("Success")
            
            info_logger.info("Check that sample_rate == SAMPLE_RATE")
            # Делаем ресемплинг до нужной частоты
            if sample_rate != SAMPLE_RATE:
                info_logger.info("sample_rate != SAMPLE_RATE")
                resampler: torchaudio.transforms.Resample = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                waveform = resampler(waveform)
            info_logger.info("END Check that sample_rate == SAMPLE_RATE")
            
            info_logger.info("Check that waveform.size(0) > 1 ")
            # Преобразуем в моно, если нужно
            if waveform.size(0) > 1:
                info_logger.info("waveform.size(0) > 1 ")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            info_logger.info("End Check that waveform.size(0) > 1 ")
            
            info_logger.info("Do normalization")
            # Нормализация
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)
            info_logger.info("End normalization")

            info_logger.info("Start noise reduction process")
            # Вычисляем спектрограмму
            spec: torch.Tensor = torchaudio.transforms.Spectrogram(
                n_fft=1024, 
                hop_length=512
            )(waveform)
            
            # Оценка шума из первых фреймов
            noise_estimate: torch.Tensor = torch.mean(spec[:, :, :10], dim=2, keepdim=True)
            
            # Спектральное вычитание
            enhanced_spec: torch.Tensor = torch.clamp(spec - noise_estimate, min=0.0)
            
            # Обратное преобразование в волновую форму
            griffin_lim: torchaudio.transforms.GriffinLim = torchaudio.transforms.GriffinLim(
                n_fft=1024, 
                hop_length=512
            )
            enhanced_waveform: torch.Tensor = griffin_lim(enhanced_spec)
            info_logger.info("End noise reduction process")
            
            # Применяем аугментацию к очищенной аудиоформе
            info_logger.info("Start applying augmentation to enhanced waveform")
            augmented_waveforms: List[torch.Tensor] = self._apply_augmentation(enhanced_waveform)
            info_logger.info("End applying augmentation to enhanced waveform")
            
            # Разбиение каждой аугментированной аудиоформы на фрагменты
            info_logger.info("Start splitting augmented waveforms into fragments")
            features_list: List[torch.Tensor] = []
            
            for aug_waveform in augmented_waveforms:
                # Разбиение на фрагменты
                fragment_length: int = int(SAMPLE_RATE * AUDIO_FRAGMENT_LENGTH)
                num_fragments: int = max(1, int(aug_waveform.size(1) / fragment_length))
                
                for i in range(num_fragments):
                    start: int = i * fragment_length
                    end: int = min(start + fragment_length, aug_waveform.size(1))
                    
                    fragment: torch.Tensor = aug_waveform[:, start:end]
                    
                    # Если фрагмент слишком короткий, дополняем его нулями
                    if end - start < fragment_length:
                        padding: torch.Tensor = torch.zeros(1, fragment_length - (end - start))
                        fragment = torch.cat([fragment, padding], dim=1)
                    
                    # Извлечение MFCC признаков
                    mfcc_transform: torchaudio.transforms.MFCC = torchaudio.transforms.MFCC(
                        sample_rate=SAMPLE_RATE,
                        n_mfcc=40,
                        log_mels=True,
                        melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
                    )
                    
                    mfcc: torch.Tensor = mfcc_transform(fragment)
                    
                    # Добавляем дельта и дельта-дельта коэффициенты
                    delta: torch.Tensor = torchaudio.functional.compute_deltas(mfcc)
                    delta2: torch.Tensor = torchaudio.functional.compute_deltas(delta)
                    
                    # Объединяем все признаки
                    combined_features: torch.Tensor = torch.cat([mfcc, delta, delta2], dim=1)
                    
                    # Делаем pad или обрезаем до фиксированной длины
                    target_length: int = MODELS_PARAMS['FEATURE_TARGET_LENGTH']
                    if combined_features.size(2) < target_length:
                        pad: torch.Tensor = torch.zeros(1, combined_features.size(1), target_length - combined_features.size(2))
                        combined_features = torch.cat([combined_features, pad], dim=2)
                    else:
                        combined_features = combined_features[:, :, :target_length]
                    
                    # Добавляем в список признаков
                    features_list.append(combined_features.squeeze(0).transpose(0, 1))
            info_logger.info("End splitting augmented waveforms into fragments")
            
            # Удаление временного файла, если он был создан
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            return features_list
            
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "_extract_features",
                "Ошибка при извлечении признаков"
            )
            # Удаление временного файла в случае ошибки, если он был создан
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
            return []
            
        finally:
            info_logger.info(f"---End extract features process in {self.module_name} model---")
    
    def train(self, audio_file: FileStorage, class_name: str) -> bool:
        """
        Обучает модель на наборе аудиофайлов и соответствующих классов.
        
        Args:
            audio_file: Аудиофайл для обучения
            class_name: Имя класса/метка для аудиофайла
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        info_logger.info(f"---Start training process in {self.module_name} model---")
        try:
            # Проверка входных данных
            if not audio_file or not class_name:
                error_logger.log_error(
                    "Пустые входные данные для обучения",
                    self.module_name,
                    "train"
                )
                return False
            
            # Проверяем, есть ли класс в словаре классов
            if class_name not in self.classes:
                # Добавляем новый класс в словарь
                class_idx = len(self.classes)
                self.classes[class_name] = class_idx
                self.index_to_class[class_idx] = class_name
                
            # Получаем индекс класса
            class_idx = self.classes[class_name]
            
            # Извлекаем признаки из аудиофайлов
            info_logger.info("Start extracting features from audio files")
            
            features: List[torch.Tensor] = self._extract_features(audio_file)
            
            if not features:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файла",
                    self.module_name,
                    "train"
                )
                return False
                
            # Создаем метки для всех фрагментов
            labels = [class_idx for _ in range(len(features))]
            
            info_logger.info("End extracting features from audio files")
                
            # Преобразуем в тензоры PyTorch
            info_logger.info("Start converting features to PyTorch tensors")
            X: torch.Tensor = torch.stack(features).to(self.device)
            y: torch.Tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
            info_logger.info("End converting features to PyTorch tensors")
            
            # Создаем модель или обновляем существующую
            self._create_or_update_model(X)
            
            # Настройка обучения
            info_logger.info("Setting up training configuration")
            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
            optimizer: optim.Adam = optim.Adam(
                self.model.parameters(), 
                lr=MODELS_PARAMS['LEARNING_RATE'], 
                weight_decay=MODELS_PARAMS['WEIGHT_DECAY']
            )
            scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=MODELS_PARAMS['SCHEDULER_FACTOR'], 
                patience=MODELS_PARAMS['SCHEDULER_PATIENCE'], 
                min_lr=0.00001
            )
            info_logger.info("Training configuration setup completed")
            
            # Создаем загрузчики данных
            info_logger.info("Creating data loaders")
            dataset = torch.utils.data.TensorDataset(X, y)
            train_size: int = int(MODELS_PARAMS['TRAIN_SPLIT'] * len(dataset))
            val_size: int = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=min(MODELS_PARAMS['BATCH_SIZE'], len(train_dataset)), 
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=min(MODELS_PARAMS['BATCH_SIZE'], len(val_dataset))
            )
            info_logger.info("Data loaders created")
            
            # Настройка раннего останова
            best_val_loss: float = float('inf')
            patience: int = MODELS_PARAMS['EARLY_STOP_PATIENCE']
            patience_counter: int = 0
            
            # Обучение
            info_logger.info("Starting model training")
            num_epochs: int = MODELS_PARAMS['EPOCHS']
            for epoch in range(num_epochs):
                # Обучение
                self.model.train()
                train_loss: float = 0.0
                train_correct: int = 0
                
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    train_correct += torch.sum(preds == labels.data).item()
                
                train_loss = train_loss / len(train_loader.dataset)
                train_acc: float = train_correct / len(train_loader.dataset)
                
                # Валидация
                self.model.eval()
                val_loss: float = 0.0
                val_correct: int = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_correct += torch.sum(preds == labels.data).item()
                
                val_loss = val_loss / len(val_loader.dataset)
                val_acc: float = val_correct / len(val_loader.dataset)
                
                # Логирование процесса обучения
                info_logger.info(f"Эпоха {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                
                # Обновление планировщика скорости обучения
                scheduler.step(val_loss)
                
                # Ранний останов
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Сохранение лучшей модели
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        # Восстановление лучшей модели
                        self.model.load_state_dict(best_model_state)
                        info_logger.info(f"Ранний останов на эпохе {epoch+1} - лучшая val_loss: {best_val_loss:.4f}")
                        break
            
            info_logger.info("Model training completed")
            
            return True
            
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "train",
                "Ошибка при обучении модели"
            )
            return False
            
        finally:
            info_logger.info(f"---End training process in {self.module_name} model---")
    
    def _predict_base(self, audio_file: FileStorage) -> Dict[str, Any]:
        """
        Базовый метод для предсказания класса из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для предсказания
            
        Returns:
            Dict[str, Any]: Результаты предсказания включая:
                - 'status': Статус операции (success/error)
                - 'features_list': Список тензоров признаков (при успехе)
                - 'error_message': Сообщение об ошибке (при ошибке)
        """
        info_logger.info(f"---Start base prediction process in {self.module_name} model---")
        try:
            # Проверка состояния модели
            if self.model is None:
                error_logger.log_error(
                    "Модель не обучена или не инициализирована",
                    self.module_name,
                    "_predict_base"
                )
                return {"status": "error", "error_message": "Модель не обучена"}
            
            # Извлекаем признаки из аудиофайла
            info_logger.info("Start extracting features from audio file")
            features_list: List[torch.Tensor] = self._extract_features(audio_file)
            info_logger.info("End extracting features from audio file")
            
            if not features_list:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файла",
                    self.module_name,
                    "_predict_base"
                )
                return {"status": "error", "error_message": "Не удалось извлечь признаки из файла"}
            
            return {"status": "success", "features_list": features_list}
            
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "_predict_base",
                "Ошибка при предсказании класса"
            )
            return {"status": "error", "error_message": str(e)}
            
        finally:
            info_logger.info(f"---End base prediction process in {self.module_name} model---")
    
    def _get_prediction_from_model(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Получает предсказание от модели для заданных признаков.
        
        Args:
            features: Тензор признаков
            
        Returns:
            Dict[str, Any]: Результаты предсказания включая:
                - 'predicted_class_index': Индекс предсказанного класса
                - 'confidence': Уверенность предсказания
        """
        info_logger.info(f"Start making prediction in {self.module_name} model")
        self.model.eval()
        
        with torch.no_grad():
            # Добавляем батч-размерность, если нужно
            if len(features.shape) == 2:
                features = features.unsqueeze(0)
                
            outputs: torch.Tensor = self.model(features)
            probabilities: torch.Tensor = torch.nn.functional.softmax(outputs, dim=1)
            
            # Находим класс с наибольшей вероятностью
            max_prob, predicted_class_index = torch.max(probabilities, 1)
            confidence: float = max_prob.item()
            predicted_class_index_int: int = predicted_class_index.item()
        
        info_logger.info(f"End making prediction in {self.module_name} model")
        
        return {
            "predicted_class_index": predicted_class_index_int,
            "confidence": confidence
        }

    def predict(self, audio_file: FileStorage, *args, **kwargs) -> Any:
        """
        Предсказывает класс из аудиофайла. Этот метод должен быть реализован в дочерних классах.
        
        Args:
            audio_file: Аудиофайл для предсказания
            *args, **kwargs: Дополнительные аргументы для конкретной реализации
            
        Returns:
            Any: Результат предсказания
        """
        raise NotImplementedError("Этот метод должен быть реализован в дочернем классе")

    # Методы, которые должны быть реализованы в дочерних классах
    def _create_or_update_model(self, features: torch.Tensor) -> None:
        """
        Создает или обновляет модель.
        
        Args:
            features: Тензор признаков для определения входной размерности
        """
        raise NotImplementedError("Этот метод должен быть реализован в дочернем классе") 