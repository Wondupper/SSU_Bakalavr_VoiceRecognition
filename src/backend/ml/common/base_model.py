import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union, Optional, Any
from werkzeug.datastructures import FileStorage
from src.backend.loggers.error_logger import error_logger
from src.backend.loggers.info_logger import info_logger
from src.backend.ml.common.audio_to_features import get_features_tensors_from_audio_for_training
from src.backend.ml.common.audio_to_features import get_features_tensors_from_audio_for_prediction
from src.backend.ml.common.train import train_one_epoch
from src.backend.ml.common.validation import calculate_batch_metrics
from src.backend.config import COMMON_MODELS_PARAMS


class BaseMLModel:
    """
    Базовая модель машинного обучения для распознавания речи.
    
    Атрибуты:
        model: Модель PyTorch
        classes: Словарь классов и соответствующих им индексов
        index_to_class: Словарь индексов и соответствующих им классов
        device: Устройство для обучения модели (CPU/GPU)
    """
    
    def __init__(self, module_name: str, model_params: Dict[str, Any]) -> None:
        """
        Инициализирует базовую модель машинного обучения.
        
        Args:
            module_name: Имя модуля для логирования
            model_params: Параметры модели
        """
        # Инициализируем атрибуты
        self.module_name = module_name
        self.model = None
        self.classes: Dict[str, int] = {}  # Словарь {класс: индекс}
        self.index_to_class: Dict[int, str] = {}  # Словарь {индекс: класс}
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем параметры модели
        self.features_target_length = model_params['FEATURE_TARGET_LENGTH']
        self.min_confidence = model_params['MIN_CONFIDENCE']
        self.train_split = COMMON_MODELS_PARAMS['TRAIN_SPLIT']
        self.early_stop_patience = COMMON_MODELS_PARAMS['EARLY_STOP_PATIENCE']
        self.batch_size = COMMON_MODELS_PARAMS['BATCH_SIZE']
        self.val_split = COMMON_MODELS_PARAMS['VAL_SPLIT']
        self.epochs = COMMON_MODELS_PARAMS['EPOCHS']
        self.patience = COMMON_MODELS_PARAMS['PATIENCE']
        self.learning_rate = COMMON_MODELS_PARAMS['LEARNING_RATE']
        self.weight_decay = COMMON_MODELS_PARAMS['WEIGHT_DECAY']
        self.scheduler_factor = COMMON_MODELS_PARAMS['SCHEDULER_FACTOR']
        self.scheduler_patience = COMMON_MODELS_PARAMS['SCHEDULER_PATIENCE']
        self.min_lr = COMMON_MODELS_PARAMS['MIN_LR']
        self.softmax_temperature = COMMON_MODELS_PARAMS['SOFTMAX_TEMPERATURE']

    @property
    def is_trained(self) -> bool:
        """
        Проверяет, обучена ли модель
        
        Returns:
            bool: True, если модель обучена, иначе False
        """
        return self.model is not None and len(self.classes) > 0
    
    def create_model(self, input_dim: int, num_classes: int) -> nn.Module:
        """
        Создает модель нейронной сети.
        Должен быть переопределен в дочерних классах.
        
        Args:
            input_dim: Размерность входных данных
            num_classes: Количество классов
            
        Returns:
            nn.Module: Модель нейронной сети
        """
        raise NotImplementedError("Метод create_model должен быть переопределен в дочернем классе")
    
    def train(self, dataset: Dict[str, FileStorage]):
        """
        Обучает модель на наборе аудиофайлов и соответствующих классов.
        
        Args:
            dataset: Словарь, где ключи - это классы, а значения - аудиофайлы
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        try:
            info_logger.info(f"{self.module_name} - Начало процесса обучения модели на наборе данных")
            info_logger.info(f"Классы: {dataset.keys()}")
            
            # Подготовка данных для обучения
            all_features: List[torch.Tensor] = []
            all_labels: List[int] = []
            
            # Обрабатываем каждый класс из набора данных
            for class_name, file in dataset.items():
                
                # Добавляем новый класс в словарь, если его нет
                if class_name not in self.classes:
                    class_idx = len(self.classes)
                    self.classes[class_name] = class_idx
                    self.index_to_class[class_idx] = class_name
                
                class_idx = self.classes[class_name]
                
                # Сброс позиции файла на начало
                file.stream.seek(0)
                
                # Извлекаем признаки из аудиофайла
                features = get_features_tensors_from_audio_for_training(file, self.features_target_length)
                
                # Добавляем признаки и метки в обучающую выборку
                all_features.extend(features)
                all_labels.extend([class_idx] * len(features))
            
            # Преобразуем в тензоры PyTorch
            X: torch.Tensor = torch.stack(all_features).to(self.device)
            y: torch.Tensor = torch.tensor(all_labels, dtype=torch.long).to(self.device)
            
            # Создаем модель, если её ещё нет
            if self.model is None:
                input_dim: int = X.size(2)
                self.model = self.create_model(input_dim, len(self.classes)).to(self.device)
            
            # Настройка обучения
            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
            self.model.criterion = criterion  # Добавляем criterion как атрибут модели для расчета метрик
            optimizer: optim.Adam = optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.scheduler_factor, 
                patience=self.scheduler_patience, 
                min_lr=self.min_lr
            )
            
            # Создаем загрузчики данных
            dataset_torch = torch.utils.data.TensorDataset(X, y)
            train_size: int = int(self.train_split * len(dataset_torch))
            val_size: int = len(dataset_torch) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset_torch, [train_size, val_size])
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=min(self.batch_size, len(train_dataset)), 
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=min(self.batch_size, len(val_dataset))
            )
            
            # Настройка раннего останова
            best_val_loss: float = float('inf')
            patience: int = self.early_stop_patience
            patience_counter: int = 0
            
            # Обучение
            num_epochs: int = self.epochs
            for epoch in range(num_epochs):
                # Обучение - выполняем одну эпоху обучения
                train_metrics = train_one_epoch(
                    self.model,
                    train_loader,
                    optimizer,
                    criterion,
                    self.device,
                    epoch,
                    num_epochs
                )
                
                # Валидация - вычисляем метрики на валидационном наборе
                val_metrics = calculate_batch_metrics(
                    self.model, 
                    val_loader, 
                    self.device, 
                    num_classes=len(self.classes),
                    epoch=epoch,
                    num_epochs=num_epochs
                )
                
                # Обновление планировщика скорости обучения
                if 'loss' in val_metrics:
                    scheduler.step(val_metrics['loss'])
                    
                    # Ранний останов
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        patience_counter = 0
                        # Сохранение лучшей модели
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            # Восстановление лучшей модели
                            self.model.load_state_dict(best_model_state)
                            info_logger.info(f"{self.module_name} - Ранний останов на эпохе {epoch+1} - лучшая val_loss: {best_val_loss:.4f}")
                            break
            
            info_logger.info(f"{self.module_name} - Обучение модели на наборе данных завершено")
            
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "train",
                "Ошибка при обучении модели"
            )
    
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
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Добавляем батч-размерность, если нужно
                if len(features.shape) == 1:
                    # Если это просто вектор логитов, добавляем размерность батча
                    features = features.unsqueeze(0)
                elif len(features.shape) == 2 and features.shape[0] == 1:
                    # Если это уже батч из одного элемента, ничего не делаем
                    pass
                
                # Перенос входных данных на устройство модели
                features = features.to(self.device)
                
                # Если это уже логиты, а не входные признаки для модели
                if features.shape[1] == len(self.classes):
                    # Применяем softmax с температурой
                    logits = features / self.softmax_temperature
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                else:
                    # Получаем вывод модели с обработкой возможных различных форматов
                    output = self.model(features)
                    
                    if isinstance(output, tuple):
                        # Берем первый элемент кортежа, если это кортеж
                        outputs = output[0]
                    else:
                        outputs = output
                    
                    # Применяем softmax с температурой
                    outputs = outputs / self.softmax_temperature
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Форматируем вывод вероятностей для более читаемого вида
                prob_dict = {}
                for idx in range(probabilities.size(1)):
                    class_name = self.index_to_class.get(idx, f"Неизвестный класс {idx}")
                    prob_value = probabilities[0, idx].item()  # Берем первый элемент батча
                    prob_dict[class_name] = f"{prob_value:.6f}"
                
                # Логируем форматированные вероятности
                info_logger.info(f"Вероятности классов: {prob_dict}")
                
                # Находим класс с наибольшей вероятностью
                max_prob, predicted_class_index = torch.max(probabilities, 1)
                confidence = max_prob.item()
                predicted_class_index_int = predicted_class_index.item()
            
            return {
                "predicted_class_index": predicted_class_index_int,
                "confidence": confidence
            }
        
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "get_prediction_from_model",
                "Ошибка при получении предсказаний"
            )
            return {}
    
    def predict(self, audio_file: FileStorage) -> str:
        """
        Метод для предсказания.
        
        Args:
            audio_file: Аудиофайл для распознавания
            
        Returns:
            str: Предсказанное имя пользователя или "unknown", если не удалось предсказать
        """
        try:
            
            # Получаем признаки из аудиофайла
            features_list = get_features_tensors_from_audio_for_prediction(audio_file, self.features_target_length)

            # Собираем батч всех фрагментов
            X = torch.stack(features_list).to(self.device)

            # Базовый упрощённый алгоритм: суммируем логиты
            self.model.eval()
            with torch.no_grad():
                # Получаем выход модели
                outputs = self.model(X)
                
                # Обрабатываем возможный кортеж
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Берем первый элемент, если это кортеж
                
                # Суммируем логиты по всем фрагментам
                summed_logits = outputs.sum(dim=0, keepdim=True)
                
                # Используем get_prediction_from_model для получения предсказания
                prediction = self.get_prediction_from_model(summed_logits)
                predicted_class_index = prediction["predicted_class_index"]
                confidence = prediction["confidence"]

                if self.module_name == 'voice_identification_model':
                    # Если уверенность хотя бы MIN_CONFIDENCE — возвращаем класс
                    if confidence >= self.min_confidence:
                        return self.index_to_class.get(predicted_class_index, "unknown")
                    else:
                        return "unknown"
                else:
                    return self.index_to_class.get(predicted_class_index)
                

        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "predict",
                "Ошибка при предсказании"
            )
            # В случае ошибки возвращаем пустую строку
            return ""
