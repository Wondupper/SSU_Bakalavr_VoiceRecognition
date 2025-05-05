import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union, Optional, Any, Set, TypeVar, Generic, cast
from werkzeug.datastructures import FileStorage
from src.backend.loggers.error_logger import error_logger
from src.backend.loggers.info_logger import info_logger
from src.backend.config import VOICE_MODEL_PARAMS
from src.backend.ml.common.features_tensors_extractor import get_features_tensors_from_audio
from src.backend.ml.common.metrics_calculation import calculate_batch_metrics
from backend.ml.common.train import train_one_epoch

class VoiceIdentificationNN(nn.Module):
    """
    Нейронная сеть для идентификации голоса на основе PyTorch.
    Архитектура упрощена для предотвращения переобучения на малых наборах данных.
    """
    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Инициализация сети для идентификации по голосу
        
        Args:
            input_dim: Размерность входных данных (features)
            num_classes: Количество классов (пользователей)
        """
        super(VoiceIdentificationNN, self).__init__()
        
        # Упрощаем архитектуру - меньше фильтров и больше регуляризации
        # Первый сверточный блок
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)  # Увеличенный дропаут
        )
        
        # Второй сверточный блок с Max Pooling для уменьшения размерности
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(48),
            nn.MaxPool1d(2),
            nn.Dropout(0.4)  # Увеличенный дропаут
        )
        
        # Третий сверточный блок
        self.conv3 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.4)  # Увеличенный дропаут
        )
        
        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Полносвязный слой для классификации - упрощен
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # Высокий дропаут для предотвращения переобучения
            nn.Linear(32, num_classes)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть
        
        Args:
            x: Входные данные [batch_size, features, time] или [batch_size, time, features]
            
        Returns:
            Предсказания модели
        """
        
        # Если данные приходят в формате [batch_size, time, features], 
        # преобразуем их в формат [batch_size, features, time]
        if x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)
            
        # Сверточные блоки
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
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
        classes: Словарь имен пользователей и соответствующих им индексов
        index_to_name: Словарь индексов и соответствующих им имен пользователей
        device: Устройство для обучения модели (CPU/GPU)
    """
    
    def __init__(self) -> None:
        """
        Инициализирует модель для идентификации по голосу.
        """
        # Инициализируем атрибуты
        self.module_name = "voice_identification_model"
        self.model = None
        self.classes: Dict[str, int] = {}  # Словарь {класс: индекс}
        self.index_to_class: Dict[int, str] = {}  # Словарь {индекс: класс}
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_trained(self) -> bool:
        """
        Проверяет, обучена ли модель
        
        Returns:
            bool: True, если модель обучена, иначе False
        """
        return self.model is not None and len(self.classes) > 0
    
    def train(self, audio_file: FileStorage, class_name: str) -> bool:
        """
        Обучает модель на наборе аудиофайлов и соответствующих классов.
        
        Args:
            audio_file: Аудиофайл для обучения
            class_name: Имя класса/метка для аудиофайла
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        try:
            
            # Проверяем, есть ли класс в словаре классов
            if class_name not in self.classes:
                # Добавляем новый класс в словарь
                class_idx = len(self.classes)
                self.classes[class_name] = class_idx
                self.index_to_class[class_idx] = class_name
                
            # Получаем индекс класса
            class_idx = self.classes[class_name]
            
            # Извлекаем признаки из аудиофайлов 
            features: List[torch.Tensor] = get_features_tensors_from_audio(audio_file, VOICE_MODEL_PARAMS['FEATURE_TARGET_LENGTH'])
            
            if not features:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файла",
                    self.module_name,
                    "train"
                )
                return False
                
            # Создаем метки для всех фрагментов
            labels = [class_idx for _ in range(len(features))]
                
            # Преобразуем в тензоры PyTorch
            X: torch.Tensor = torch.stack(features).to(self.device)
            y: torch.Tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
            
            # Создаем модель или обновляем существующую
            if self.model is None:
                input_dim: int = X.size(2)
                model_class = self.__class__.__orig_bases__[0].__args__[0]
                self.model = model_class(input_dim, len(self.classes)).to(self.device)
            else:
                # Расширяем выходной слой, если добавился новый класс
                new_num_classes = len(self.classes)
                # Последний линейный слой в fc должен быть nn.Linear(in_f, out_f)
                old_fc = self.model.fc[-1]
                old_out = old_fc.out_features
                if new_num_classes != old_out:
                    new_fc = nn.Linear(old_fc.in_features, new_num_classes).to(self.device)
                    with torch.no_grad():
                        # Копируем старые веса/биасы в первые столбцы
                        new_fc.weight[:old_out] = old_fc.weight
                        new_fc.bias[:old_out]   = old_fc.bias
                    self.model.fc[-1] = new_fc
            
            # Настройка обучения
            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
            self.model.criterion = criterion  # Добавляем criterion как атрибут модели для расчета метрик
            optimizer: optim.Adam = optim.Adam(
                self.model.parameters(), 
                lr=VOICE_MODEL_PARAMS['LEARNING_RATE'], 
                weight_decay=VOICE_MODEL_PARAMS['WEIGHT_DECAY']
            )
            scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=VOICE_MODEL_PARAMS['SCHEDULER_FACTOR'], 
                patience=VOICE_MODEL_PARAMS['SCHEDULER_PATIENCE'], 
                min_lr=0.00001
            )
            
            # Создаем загрузчики данных
            dataset = torch.utils.data.TensorDataset(X, y)
            train_size: int = int(VOICE_MODEL_PARAMS['TRAIN_SPLIT'] * len(dataset))
            val_size: int = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=min(VOICE_MODEL_PARAMS['BATCH_SIZE'], len(train_dataset)), 
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=min(VOICE_MODEL_PARAMS['BATCH_SIZE'], len(val_dataset))
            )
            
            # Настройка раннего останова
            best_val_loss: float = float('inf')
            patience: int = VOICE_MODEL_PARAMS['EARLY_STOP_PATIENCE']
            patience_counter: int = 0
            
            # Обучение
            num_epochs: int = VOICE_MODEL_PARAMS['EPOCHS']
            for epoch in range(num_epochs):
                # Обучение - выполняем одну эпоху обучения
                train_one_epoch(
                    self.model,
                    train_loader,
                    optimizer,
                    criterion,
                    self.device,
                    epoch=epoch,
                    num_epochs=num_epochs
                )
                
                # Валидация - вычисляем метрики на валидационном наборе
                self.model.eval()
                val_loss = calculate_batch_metrics(
                    self.model, 
                    val_loader, 
                    self.device, 
                    num_classes=len(self.classes),
                    epoch=epoch,
                    num_epochs=num_epochs
                )
                
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
            
            
            return True
            
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "train",
                "Ошибка при обучении модели"
            )
            return False
     
    def get_prediction_from_model(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Получает предсказание от модели для заданных признаков.
        
        Args:
            features: Тензор признаков
            
        Returns:
            Dict[str, Any]: Результаты предсказания включая:
                - 'predicted_class_index': Индекс предсказанного класса
                - 'confidence': Уверенность предсказания
        """
        self.model.eval()
        
        with torch.no_grad():
            # Добавляем батч-размерность, если нужно
            if len(features.shape) == 2:
                features = features.unsqueeze(0)
                
            # Перенос входных данных на устройство модели
            features = features.to(self.device)
                
            outputs: torch.Tensor = self.model(features)
            probabilities: torch.Tensor = torch.nn.functional.softmax(outputs, dim=1)
            
            # Находим класс с наибольшей вероятностью
            max_prob, predicted_class_index = torch.max(probabilities, 1)
            confidence: float = max_prob.item()
            predicted_class_index_int: int = predicted_class_index.item()
        
        return {
            "predicted_class_index": predicted_class_index_int,
            "confidence": confidence
        }
    
    def predict(self, audio_file: FileStorage) -> str:
        """
        Расшиернный метод для предсказания класса из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для распознавания
            
        Returns:
            str: Предсказанный класс или "unknown", если не удалось предсказать
        """
        try:
            features_list = get_features_tensors_from_audio(audio_file)
            if not features_list:
                # … логирование …
                return "unknown"

            # Собираем батч всех фрагментов
            X = torch.stack(features_list).to(self.device)

            # Базовый упрощённый алгоритм: суммируем логиты
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)                       # [num_fragments, num_classes]
                summed_logits = outputs.sum(dim=0, keepdim=True)  # [1, num_classes]
                probs = torch.softmax(summed_logits, dim=1)[0]    # [num_classes]
                best_idx = torch.argmax(probs).item()
                best_conf = probs[best_idx].item()

            # Если уверенность хотя бы MIN_AVG_CONFIDENCE — возвращаем класс
            if best_conf >= VOICE_MODEL_PARAMS['MIN_AVG_CONFIDENCE']:
                return self.index_to_class.get(best_idx, "unknown")
            else:
                return "unknown"

        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "predict",
                "Ошибка при предсказании"
            )
            return "unknown"

            
        finally:
            info_logger.info(f"---End extended prediction process in {self.module_name} model---")

    
