import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union, Optional, Any
from werkzeug.datastructures import FileStorage
from src.backend.loggers.error_logger import error_logger
from src.backend.loggers.info_logger import info_logger
from src.backend.ml.common.audio_to_features import get_features_tensors_from_audio
from src.backend.ml.common.train import train_one_epoch
from src.backend.ml.common.validation import calculate_batch_metrics


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
        self.train_split = model_params['TRAIN_SPLIT']
        self.early_stop_patience = model_params['EARLY_STOP_PATIENCE']
        self.batch_size = model_params['BATCH_SIZE']
        self.val_split = model_params['VAL_SPLIT']
        self.epochs = model_params['EPOCHS']
        self.patience = model_params['PATIENCE']
        self.learning_rate = model_params['LEARNING_RATE']
        self.weight_decay = model_params['WEIGHT_DECAY']
        self.scheduler_factor = model_params['SCHEDULER_FACTOR']
        self.scheduler_patience = model_params['SCHEDULER_PATIENCE']
        self.min_lr = model_params['MIN_LR']

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
    
    def train(self, dataset: Dict[str, FileStorage]) -> bool:
        """
        Обучает модель на наборе аудиофайлов и соответствующих классов.
        
        Args:
            dataset: Словарь, где ключи - это классы, а значения - аудиофайлы
            
        Returns:
            bool: Успешно ли завершилось обучение
        """
        try:
            info_logger.info(f"{self.module_name} - Начало процесса обучения модели на наборе данных")
            
            # Проверяем, что набор данных не пустой
            if not dataset:
                error_logger.log_error(
                    "Набор данных для обучения пуст",
                    self.module_name,
                    "train"
                )
                return False
            
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
                features = get_features_tensors_from_audio(file, self.features_target_length)
                
                if not features:
                    error_logger.log_error(
                        f"Не удалось извлечь признаки из файла {file.filename} для класса {class_name}",
                        self.module_name,
                        "train"
                    )
                    continue
                
                # Добавляем признаки и метки в обучающую выборку
                all_features.extend(features)
                all_labels.extend([class_idx] * len(features))
                
                info_logger.info(f"{self.module_name} - Данные для класса {class_name} из файла {file.filename} успешно загружены")
            
            if not all_features:
                error_logger.log_error(
                    "Не удалось извлечь признаки ни из одного аудиофайла",
                    self.module_name,
                    "train"
                )
                return False
            
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
        try:
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
        
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "get_prediction_from_model",
                "Ошибка при получении предсказаний"
            )
            return {
                "predicted_class_index": -1,
                "confidence": 0.0
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
            if best_conf >= self.min_confidence:
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
