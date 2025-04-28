import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import random
from backend.api.error_logger import error_logger
from backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH

class VoiceIdentificationNN(nn.Module):
    """
    Нейронная сеть для идентификации голоса на основе PyTorch.
    Использует свёрточную архитектуру с дилатацией.
    """
    def __init__(self, input_dim, num_classes):
        """
        Инициализация сети для идентификации по голосу
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (пользователей)
        """
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
        
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Args:
            x: Входные данные [batch_size, features, time]
            
        Returns:
            Предсказания модели
        """
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
    
    def __init__(self):
        """
        Инициализирует модель для идентификации по голосу.
        """
        # Инициализируем атрибуты
        self.model = None
        self.classes = []
        self.is_trained = False
        self.is_training = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
    def _extract_features(self, audio_file):
        """
        Извлекает признаки из аудиофайла с помощью torchaudio
        
        Args:
            audio_file: Файл аудио (объект FileStorage Flask)
            
        Returns:
            Список тензоров признаков для каждого фрагмента
        """
        try:
            # Сохраняем аудиофайл во временный файл
            temp_filename = f"temp_audio_{random.randint(1000, 9999)}.wav"
            audio_file.save(temp_filename)
            
            # Загружаем аудио и делаем ресемплинг до нужной частоты
            waveform, sample_rate = torchaudio.load(temp_filename)
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # Преобразуем в моно, если нужно
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Нормализация
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)
            
            # Удаление шума с помощью спектрального вычитания
            # Вычисляем спектрограмму
            spec = torchaudio.transforms.Spectrogram(
                n_fft=1024, 
                hop_length=512
            )(waveform)
            
            # Оценка шума из первых фреймов
            noise_estimate = torch.mean(spec[:, :, :10], dim=2, keepdim=True)
            
            # Спектральное вычитание
            enhanced_spec = torch.clamp(spec - noise_estimate, min=0.0)
            
            # Обратное преобразование в волновую форму
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=1024, 
                hop_length=512
            )
            enhanced_waveform = griffin_lim(enhanced_spec)
            
            # Разбиение на фрагменты
            fragment_length = int(SAMPLE_RATE * AUDIO_FRAGMENT_LENGTH)
            num_fragments = max(1, int(enhanced_waveform.size(1) / fragment_length))
            
            features_list = []
            for i in range(num_fragments):
                start = i * fragment_length
                end = min(start + fragment_length, enhanced_waveform.size(1))
                
                fragment = enhanced_waveform[:, start:end]
                
                # Если фрагмент слишком короткий, дополняем его нулями
                if end - start < fragment_length:
                    padding = torch.zeros(1, fragment_length - (end - start))
                    fragment = torch.cat([fragment, padding], dim=1)
                
                # Извлечение MFCC признаков
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=SAMPLE_RATE,
                    n_mfcc=40,
                    log_mels=True,
                    melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
                )
                mfcc = mfcc_transform(fragment)
                
                # Добавляем дельта и дельта-дельта коэффициенты
                delta = torchaudio.functional.compute_deltas(mfcc)
                delta2 = torchaudio.functional.compute_deltas(delta)
                
                # Объединяем все признаки
                combined_features = torch.cat([mfcc, delta, delta2], dim=1)
                
                # Делаем pad или обрезаем до фиксированной длины
                target_length = 200  # Подбираем нужное значение
                if combined_features.size(2) < target_length:
                    pad = torch.zeros(1, combined_features.size(1), target_length - combined_features.size(2))
                    combined_features = torch.cat([combined_features, pad], dim=2)
                else:
                    combined_features = combined_features[:, :, :target_length]
                
                # Добавляем в список признаков
                features_list.append(combined_features.squeeze(0).transpose(0, 1))
            
            # Удаление временного файла
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            return features_list
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "_extract_features",
                "Ошибка при извлечении признаков"
            )
            # Удаление временного файла в случае ошибки
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return []
    
    def train(self, audio_files, names):
        """
        Обучает модель на наборе аудиофайлов и соответствующих имен.
        
        Args:
            audio_files: Список аудиофайлов для обучения
            names: Список имен пользователей для каждого аудиофайла
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        try:
            # Проверка входных данных
            if not audio_files or not names:
                error_logger.log_error(
                    "Пустые входные данные для обучения",
                    "voice_identification",
                    "train"
                )
                return False
                
            if len(audio_files) != len(names):
                error_logger.log_error(
                    "Количество аудиофайлов не соответствует количеству имен",
                    "voice_identification",
                    "train"
                )
                return False
                
            # Устанавливаем флаг, что идет обучение
            self.is_training = True
            
            # Обновляем список классов
            unique_names = sorted(list(set(names)))
            
            # Создаем отображение имен на индексы
            name_to_index = {name: i for i, name in enumerate(unique_names)}
            
            # Извлекаем признаки из аудиофайлов
            all_features = []
            all_labels = []
            
            for audio_file, name in zip(audio_files, names):
                features = self._extract_features(audio_file)
                if not features:
                    error_logger.log_error(
                        f"Не удалось извлечь признаки из файла для пользователя {name}",
                        "voice_identification",
                        "train"
                    )
                    continue
                
                for feature in features:
                    all_features.append(feature)
                    all_labels.append(name_to_index[name])
            
            if not all_features:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файлов",
                    "voice_identification",
                    "train"
                )
                self.is_training = False
                return False
            
            # Преобразуем в тензоры PyTorch
            X = torch.stack(all_features).to(self.device)
            y = torch.tensor(all_labels, dtype=torch.long).to(self.device)
            
            # Проверка, создана ли модель и соответствует ли она текущим классам
            if self.model is None or len(self.classes) != len(unique_names) or not all(x in self.classes for x in unique_names):
                # Обновляем список классов
                self.classes = unique_names
                
                # Создаем новую модель
                input_dim = X.size(2)
                self.model = VoiceIdentificationNN(input_dim, len(self.classes)).to(self.device)
            
            # Настройка обучения
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001)
            
            # Создаем загрузчики данных
            dataset = torch.utils.data.TensorDataset(X, y)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
            
            # Настройка раннего останова
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            # Обучение
            num_epochs = 100
            for epoch in range(num_epochs):
                # Обучение
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    train_correct += torch.sum(preds == labels.data)
                
                train_loss = train_loss / len(train_loader.dataset)
                train_acc = train_correct.double() / len(train_loader.dataset)
                
                # Валидация
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_correct += torch.sum(preds == labels.data)
                
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_correct.double() / len(val_loader.dataset)
                
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
                        break
            
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
    
    def predict(self, audio_file):
        """
        Идентифицирует пользователя по голосу из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для идентификации
            
        Returns:
            str: Имя пользователя или "unknown", если не удалось идентифицировать
        """
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
            features_list = self._extract_features(audio_file)
            
            if not features_list:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файла",
                    "voice_identification",
                    "predict"
                )
                return "unknown"
            
            # Преобразуем в тензоры PyTorch
            X = torch.stack(features_list).to(self.device)
            
            # Предсказание
            self.model.eval()
            
            # Голосование по результатам фрагментов
            vote_counts = {}
            confidence_sums = {}
            
            with torch.no_grad():
                for feature in X:
                    feature = feature.unsqueeze(0)  # Добавляем измерение батча
                    outputs = self.model(feature)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Находим класс с наибольшей вероятностью
                    max_prob, predicted_class_index = torch.max(probabilities, 1)
                    confidence = max_prob.item()
                    predicted_class_index = predicted_class_index.item()
                    
                    # Если уверенность выше порога, идентифицируем пользователя
                    if confidence >= 0.8:
                        label = self.classes[predicted_class_index]
                    else:
                        label = "unknown"
                    
                    if label not in vote_counts:
                        vote_counts[label] = 0
                        confidence_sums[label] = 0
                    
                    vote_counts[label] += 1
                    confidence_sums[label] += confidence
            
            # Находим метку с наибольшим количеством голосов
            if vote_counts:
                identity = max(vote_counts.keys(), key=lambda k: vote_counts[k])
                avg_confidence = confidence_sums[identity] / vote_counts[identity]
                
                # Если средняя уверенность низкая или это "unknown", считаем неизвестным
                if avg_confidence < 0.6 or identity == "unknown":
                    return "unknown"
                    
                return identity
            else:
                return "unknown"
            
        except Exception as e:
            error_logger.log_exception(
                e,
                "voice_identification",
                "predict",
                "Ошибка при идентификации пользователя"
            )
            
            return "unknown"
