import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union, Optional, Any, Set, TypeVar, Generic, cast
from werkzeug.datastructures import FileStorage
from src.backend.loggers.error_logger import error_logger
from src.backend.loggers.info_logger import info_logger
from src.backend.config import MODELS_PARAMS
from src.backend.ml.utils.augmentation import apply_augmentation
from src.backend.ml.utils.features_tensors_extractor import get_features_tensors_from_audio

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
        # Инициализируем атрибуты
        self.model: Optional[T] = None
        self.classes: Dict[str, int] = {}  # Словарь {класс: индекс}
        self.index_to_class: Dict[int, str] = {}  # Словарь {индекс: класс}
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module_name = module_name
        
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
        return apply_augmentation(waveform, self.module_name)
    
    def _get_features_tensors_from_audio(self, audio_file: FileStorage) -> List[torch.Tensor]:
        """
        Извлекает признаки из аудиофайла с помощью torchaudio
        
        Args:
            audio_file: Файл аудио (объект FileStorage Flask)
            
        Returns:
            Список тензоров признаков для каждого фрагмента
        """
        return get_features_tensors_from_audio(audio_file, self.module_name)
    
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
            features: List[torch.Tensor] = self._get_features_tensors_from_audio(audio_file)
            
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
            
            # Создаем загрузчики данных
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
            
            # Настройка раннего останова
            best_val_loss: float = float('inf')
            patience: int = MODELS_PARAMS['EARLY_STOP_PATIENCE']
            patience_counter: int = 0
            
            # Обучение
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
            
            
            return True
            
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "train",
                "Ошибка при обучении модели"
            )
            return False
    
    
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
            features_list = self._get_features_tensors_from_audio(audio_file)
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
            if best_conf >= MODELS_PARAMS['MIN_AVG_CONFIDENCE']:
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
