import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import random
import io
from typing import List, Dict, Tuple, Union, Optional, Any, Set
from werkzeug.datastructures import FileStorage
from backend.api.error_logger import error_logger
from backend.api.info_logger import info_logger
from backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH, AUGMENTATION, VOICE_ID_MODEL

class VoiceIdentificationNN(nn.Module):
    """
    Нейронная сеть для идентификации голоса на основе PyTorch.
    Использует свёрточную архитектуру с дилатацией.
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Инициализация сети для идентификации по голосу
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (пользователей)
        """
        info_logger.info("---Start building VoiceIdentification model---")
        super(VoiceIdentificationNN, self).__init__()
        
        # Первый сверточный блок
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Второй сверточный блок с большей дилатацией
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Третий сверточный блок с ещё большей дилатацией
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Четвертый сверточный блок с шагом для уменьшения размерности
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Пятый сверточный блок для извлечения высокоуровневых признаков
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Полносвязный слой для классификации
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        info_logger.info("---Finish building VoiceIdentification model---")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть
        
        Args:
            x: Входные данные [batch_size, features, time]
            
        Returns:
            Предсказания модели
        """
        info_logger.info("---Start forwarding VoiceIdentification model---")
        # Сверточные блоки
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Глобальный пулинг
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Полносвязный слой
        x = self.fc(x)
        info_logger.info("---Finish forwarding VoiceIdentification model---")
        return x

class VoiceIdentificationModel:
    """
    Модель для идентификации пользователя по голосу.
    
    Атрибуты:
        model: Модель PyTorch для идентификации по голосу
        classes: Список имен пользователей, которые модель может распознать
        is_trained: Флаг, указывающий, обучена ли модель
        is_training: Флаг, указывающий, идет ли процесс обучения
        device: Устройство для обучения модели (CPU/GPU)
    """
    
    def __init__(self) -> None:
        """
        Инициализирует модель для идентификации по голосу.
        """
        info_logger.info("---Start initializing VoiceIdentification model---")
        # Инициализируем атрибуты
        self.model: Optional[VoiceIdentificationNN] = None
        self.classes: Set[str] = {}
        self.is_trained: bool = False
        self.is_training: bool = False
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        info_logger.info("---Finish initializing VoiceIdentification model---")
        
    def _apply_augmentation(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Применяет аугментацию к аудиофайлу для расширения обучающей выборки.
        
        Args:
            waveform: Тензор аудио [channels, time]
            
        Returns:
            Список аугментированных аудиофайлов
        """
        info_logger.info("---Start augmentation process in VoiceIdentification model---")
        try:
            augmented_waveforms: List[torch.Tensor] = [waveform]  # Добавляем оригинальное аудио
            
            new_augmented_waveforms: List[torch.Tensor] = []
            info_logger.info("Start augmentation process (FAST_SPEED) in VoiceIdentification model")
            # 1. Ускорение аудио (time stretching)
            for speed in AUGMENTATION['FAST_SPEEDS']:
                effects: List[List[str]] = [
                    ["speed", str(speed)],
                    ["rate", str(SAMPLE_RATE)]
                ]
                aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, SAMPLE_RATE, effects)
                new_augmented_waveforms.append(aug_waveform)
            info_logger.info("End augmentation process (FAST_SPEED) in VoiceIdentification model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info("Start augmentation process (SLOW_SPEED) in VoiceIdentification model")
            # 2. Замедление аудио
            for speed in AUGMENTATION['SLOW_SPEEDS']:
                effects = [
                    ["speed", str(speed)],
                    ["rate", str(SAMPLE_RATE)]
                ]
                aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, SAMPLE_RATE, effects)
                new_augmented_waveforms.append(aug_waveform)
            info_logger.info("End augmentation process (SLOW_SPEED) in VoiceIdentification model")

            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info("Start augmentation process (REVERBIRATION) in VoiceIdentification model")
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
            info_logger.info("End augmentation process (REVERBIRATION) in VoiceIdentification model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info("Start augmentation process (TIME_MASKING) in VoiceIdentification model")
            # 4. Маскирование по времени (Time Masking)
            for mask_param in AUGMENTATION['MASK_PARAMS']:
                mask_waveform: torch.Tensor = waveform.clone()
                time_mask_samples: int = int(mask_param * waveform.size(1))
                if time_mask_samples > 0:
                    mask_start: int = random.randint(0, waveform.size(1) - time_mask_samples)
                    mask_waveform[:, mask_start:mask_start + time_mask_samples] = 0
                    new_augmented_waveforms.append(mask_waveform)
            info_logger.info("End augmentation process (TIME_MASKING) in VoiceIdentification model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
            new_augmented_waveforms = []
            
            info_logger.info("Start augmentation process (EHO_ADDING) in VoiceIdentification model")
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
            info_logger.info("End augmentation process (EHO_ADDING) in VoiceIdentification model")
            
            augmented_waveforms.extend(new_augmented_waveforms)
                
            return augmented_waveforms
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "_apply_augmentation",
                "Ошибка при аугментации аудио"
            )
            # В случае ошибки возвращаем только оригинальное аудио
            return [waveform]
            
        finally:
            info_logger.info("---End augmentation process in VoiceIdentification model---")
    
    def _extract_features(self, audio_file: FileStorage) -> List[torch.Tensor]:
        """
        Извлекает признаки из аудиофайла с помощью torchaudio
        
        Args:
            audio_file: Файл аудио (объект FileStorage Flask)
            
        Returns:
            Список тензоров признаков для каждого фрагмента
        """
        info_logger.info("---Start extract features process in VoiceIdentification model---")
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
                temp_filename = f"temp_audio_id_{temp_id}.wav"
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
            
            # Удаление шума с помощью спектрального вычитания
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
                    target_length: int = VOICE_ID_MODEL['FEATURE_TARGET_LENGTH']
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
                "voice_identification",
                "_extract_features",
                "Ошибка при извлечении признаков"
            )
            # Удаление временного файла в случае ошибки, если он был создан
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)
            return []
            
        finally:
            info_logger.info("---End extract features process in VoiceIdentification model---")
    
    def train(self, audio_file: FileStorage, name: str) -> bool:
        """
        Обучает модель на наборе аудиофайлов и соответствующих имен.
        
        Args:
            audio_file: Аудиофайлов для обучения
            names: Имен пользователя для аудиофайла
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        info_logger.info("---Start training process in VoiceIdentification model---")
        try:
            # Проверка входных данных
            if not audio_file or not name:
                error_logger.log_error(
                    "Пустые входные данные для обучения",
                    "voice_identification",
                    "train"
                )
                return False
                
            # Устанавливаем флаг, что идет обучение
            self.is_training = True
            
            # Извлекаем признаки из аудиофайлов
            info_logger.info("Start extracting features from audio files")
            all_features: List[torch.Tensor] = []
            all_labels: List[int] = []
            
            features: List[torch.Tensor] = self._extract_features(audio_file)
            if not features:
                error_logger.log_error(
                    f"Не удалось извлечь признаки из файла для пользователя {name}",
                    "voice_identification",
                    "train"
                )
            
            for feature in features:
                all_features.append(feature)
                all_labels.append(name)
            info_logger.info("End extracting features from audio files")
            
            if not all_features:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файлов",
                    "voice_identification",
                    "train"
                )
                self.is_training = False
                return False
            
            # Преобразуем в тензоры PyTorch
            info_logger.info("Start converting features to PyTorch tensors")
            X: torch.Tensor = torch.stack(all_features).to(self.device)
            y: torch.Tensor = torch.tensor(all_labels, dtype=torch.long).to(self.device)
            info_logger.info("End converting features to PyTorch tensors")
            
            # Проверка, создана ли модель и соответствует ли она текущим классам
            if self.model is None or not name in self.classes:
                info_logger.info("Creating new VoiceIdentification model")
                # Обновляем список классов
                self.classes.add(name)
                
                # Создаем новую модель
                input_dim: int = X.size(2)
                self.model = VoiceIdentificationNN(input_dim, len(self.classes)).to(self.device)
                info_logger.info("New VoiceIdentification model created")
            
            # Настройка обучения
            info_logger.info("Setting up training configuration")
            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
            optimizer: optim.Adam = optim.Adam(
                self.model.parameters(), 
                lr=VOICE_ID_MODEL['LEARNING_RATE'], 
                weight_decay=VOICE_ID_MODEL['WEIGHT_DECAY']
            )
            scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=VOICE_ID_MODEL['SCHEDULER_FACTOR'], 
                patience=VOICE_ID_MODEL['SCHEDULER_PATIENCE'], 
                min_lr=0.00001
            )
            info_logger.info("Training configuration setup completed")
            
            # Создаем загрузчики данных
            info_logger.info("Creating data loaders")
            dataset = torch.utils.data.TensorDataset(X, y)
            train_size: int = int(VOICE_ID_MODEL['TRAIN_SPLIT'] * len(dataset))
            val_size: int = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=min(VOICE_ID_MODEL['BATCH_SIZE'], len(train_dataset)), 
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=min(VOICE_ID_MODEL['BATCH_SIZE'], len(val_dataset))
            )
            info_logger.info("Data loaders created")
            
            # Настройка раннего останова
            best_val_loss: float = float('inf')
            patience: int = VOICE_ID_MODEL['EARLY_STOP_PATIENCE']
            patience_counter: int = 0
            
            # Обучение
            info_logger.info("Starting model training")
            num_epochs: int = VOICE_ID_MODEL['EPOCHS']
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
            
            # Устанавливаем флаг, что модель обучена
            self.is_trained = True
            
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
            
        finally:
            info_logger.info("---End training process in VoiceIdentification model---")
    
    def predict(self, audio_file: FileStorage) -> str:
        """
        Идентифицирует пользователя по голосу из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для идентификации
            
        Returns:
            str: Имя пользователя или "unknown", если не удалось идентифицировать
        """
        info_logger.info("---Start prediction process in VoiceIdentification model---")
        try:
            # Проверка состояния модели
            if not self.is_trained or self.model is None:
                error_logger.log_error(
                    "Модель не обучена или не инициализирована",
                    "voice_identification",
                    "predict"
                )
                return "unknown"
            
            # Извлекаем признаки из аудиофайла
            info_logger.info("Start extracting features from audio file")
            features_list: List[torch.Tensor] = self._extract_features(audio_file)
            info_logger.info("End extracting features from audio file")
            
            if not features_list:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файла",
                    "voice_identification",
                    "predict"
                )
                return "unknown"
            
            # Преобразуем в тензоры PyTorch
            info_logger.info("Start converting features to PyTorch tensors")
            X: torch.Tensor = torch.stack(features_list).to(self.device)
            info_logger.info("End converting features to PyTorch tensors")
            
            # Предсказание
            info_logger.info("Start making prediction")
            self.model.eval()
            
            # Голосование по результатам фрагментов
            vote_counts: Dict[str, int] = {}
            confidence_sums: Dict[str, float] = {}
            
            with torch.no_grad():
                for feature in X:
                    feature = feature.unsqueeze(0)  # Добавляем измерение батча
                    outputs: torch.Tensor = self.model(feature)
                    probabilities: torch.Tensor = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Находим класс с наибольшей вероятностью
                    max_prob, predicted_class_index = torch.max(probabilities, 1)
                    confidence: float = max_prob.item()
                    predicted_class_index_int: int = predicted_class_index.item()
                    
                    # Если уверенность выше порога, идентифицируем пользователя
                    if confidence >= VOICE_ID_MODEL['MIN_CONFIDENCE']:
                        label: str = self.classes[predicted_class_index_int]
                    else:
                        label = "unknown"
                    
                    if label not in vote_counts:
                        vote_counts[label] = 0
                        confidence_sums[label] = 0.0
                    
                    vote_counts[label] += 1
                    confidence_sums[label] += confidence
            
            # Находим метку с наибольшим количеством голосов
            if vote_counts:
                identity: str = max(vote_counts.keys(), key=lambda k: vote_counts[k])
                avg_confidence: float = confidence_sums[identity] / vote_counts[identity]
                
                # Если средняя уверенность низкая или это "unknown", считаем неизвестным
                if avg_confidence < VOICE_ID_MODEL['MIN_AVG_CONFIDENCE'] or identity == "unknown":
                    info_logger.info("Prediction result: unknown (low confidence)")
                    return "unknown"
                    
                info_logger.info(f"Prediction result: {identity} (confidence: {avg_confidence:.4f})")
                return identity
            else:
                info_logger.info("Prediction result: unknown (no votes)")
                return "unknown"
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "predict",
                "Ошибка при идентификации пользователя"
            )
            
            return "unknown"
            
        finally:
            info_logger.info("---End prediction process in VoiceIdentification model---")
