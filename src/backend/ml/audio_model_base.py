import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union, Optional, Any, Set, TypeVar, Generic, cast
from werkzeug.datastructures import FileStorage
from backend.api.error_logger import error_logger
from backend.api.info_logger import info_logger
from backend.config import MODELS_PARAMS
from backend.ml.utils.augmentation import apply_augmentation
from backend.ml.utils.features_tensors_extractor import get_features_tensors_from_audio

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
                # Создаем новую модель
                input_dim: int = X.size(2)
                model_class = self.__class__.__orig_bases__[0].__args__[0]
                self.model = model_class(input_dim, len(self.classes)).to(self.device)
            
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

    def predict_extended(self, audio_file: FileStorage) -> str:
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
                error_logger.log_error(
                    "Не удалось извлечь признаки из файла",
                    self.module_name,
                    "predict"
                )
                return "unknown"
            
            # Расширенный алгоритм голосования (для идентификации по голосу)
            # Преобразуем в тензоры PyTorch
            X: torch.Tensor = torch.stack(features_list).to(self.device)
            
            # Улучшенный алгоритм голосования по результатам фрагментов
            vote_counts: Dict[str, int] = {}
            confidence_sums: Dict[str, float] = {}
            prediction_confidences: Dict[str, List[float]] = {}  # Для хранения всех уверенностей по классам
            
            with torch.no_grad():
                # Получаем все предсказания для всех фрагментов
                all_predictions: List[Dict[str, Any]] = []
                for feature in X:
                    prediction_result = self._get_prediction_from_model(feature)
                    confidence = prediction_result["confidence"]
                    predicted_class_index_int = prediction_result["predicted_class_index"]
                    
                    # Если класс распознан с достаточной уверенностью, добавляем в результаты
                    if predicted_class_index_int in self.index_to_class:
                        label: str = self.index_to_class[predicted_class_index_int]
                    else:
                        label = "unknown"
                    
                    # Сохраняем результат предсказания
                    all_predictions.append({
                        "label": label,
                        "confidence": confidence,
                        "index": predicted_class_index_int
                    })
                    
                    # Обновляем статистику для каждого класса
                    if label not in prediction_confidences:
                        prediction_confidences[label] = []
                    prediction_confidences[label].append(confidence)
            
            # Если нет предсказаний, возвращаем "unknown"
            if not all_predictions:
                info_logger.info("No predictions made")
                return "unknown"
            
            # Обрабатываем предсказания с учетом порога уверенности
            for pred in all_predictions:
                label = pred["label"]
                confidence = pred["confidence"]
                
                # Учитываем только предсказания с уверенностью выше порога
                if confidence >= MODELS_PARAMS['MIN_CONFIDENCE']:
                    if label not in vote_counts:
                        vote_counts[label] = 0
                        confidence_sums[label] = 0.0
                    
                    vote_counts[label] += 1
                    confidence_sums[label] += confidence
            
            # Если ни одно предсказание не прошло порог уверенности
            if not vote_counts:
                info_logger.info("No predictions passed confidence threshold")
                return "unknown"
            
            # Вычисляем статистические характеристики для каждого класса
            class_stats: Dict[str, Dict[str, float]] = {}
            for label, confidences in prediction_confidences.items():
                if not confidences:
                    continue
                    
                # Вычисляем среднюю и медианную уверенность
                avg_conf = sum(confidences) / len(confidences)
                median_conf = sorted(confidences)[len(confidences) // 2]
                
                # Вычисляем долю предсказаний с уверенностью выше порога
                confident_preds = sum(1 for c in confidences if c >= MODELS_PARAMS['MIN_CONFIDENCE'])
                confidence_ratio = confident_preds / len(confidences) if confidences else 0
                
                class_stats[label] = {
                    "avg_confidence": avg_conf,
                    "median_confidence": median_conf,
                    "confidence_ratio": confidence_ratio,
                    "vote_count": vote_counts.get(label, 0),
                    "total_samples": len(confidences)
                }
                
                info_logger.info(f"Class {label} stats: {class_stats[label]}")
            
            # Определение лучшего класса на основе голосования и статистик
            if "unknown" in class_stats:
                # Если "unknown" имеет высокий рейтинг, исключаем его из рассмотрения
                del class_stats["unknown"]
            
            if not class_stats:
                info_logger.info("No valid classes after statistics analysis")
                return "unknown"
            
            # Находим класс с наивысшим соотношением уверенных предсказаний
            best_class = max(class_stats.keys(), 
                            key=lambda k: (class_stats[k]["confidence_ratio"],
                                            class_stats[k]["avg_confidence"]))
            
            # Проверка финального результата
            if (class_stats[best_class]["avg_confidence"] >= MODELS_PARAMS['MIN_AVG_CONFIDENCE'] and
                class_stats[best_class]["confidence_ratio"] >= 0.5):
                info_logger.info(f"Prediction result: {best_class} with avg confidence: {class_stats[best_class]['avg_confidence']:.4f}")
                return best_class
            else:
                info_logger.info(f"Prediction rejected: {best_class} (insufficient confidence statistics)")
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

    
    def predict(self, audio_file: FileStorage) -> str:
        """
        Простой метод для предсказания класса из аудиофайла.
        
        Args:
            audio_file: Аудиофайл для распознавания
            
        Returns:
            str: Предсказанный класс или "unknown", если не удалось предсказать
        """
        try:
            features_list = self._get_features_tensors_from_audio(audio_file)
            
            if not features_list:
                error_logger.log_error(
                    "Не удалось извлечь признаки из файла",
                    self.module_name,
                    "predict"
                )
                return "unknown"
            
            # Простой алгоритм (для распознавания эмоций) - используем только первый элемент из features_list
            features = features_list[0]
            
            # Получаем предсказание модели
            prediction_result = self._get_prediction_from_model(features)
            confidence = prediction_result["confidence"]
            predicted_class_index_int = prediction_result["predicted_class_index"]
            
            # Если уверенность выше порога, распознаем класс
            if confidence >= MODELS_PARAMS['MIN_CONFIDENCE']:
                # Получаем класс по индексу
                if predicted_class_index_int in self.index_to_class:
                    result: str = self.index_to_class[predicted_class_index_int]
                else:
                    result = "unknown"
            else:
                result = "unknown"
            
            info_logger.info(f"Recognized class: {result} with confidence: {confidence}")
            
            # Возвращаем распознанный класс
            return result
                
        except Exception as e:
            error_logger.log_exception(
                e,
                self.module_name,
                "predict",
                "Ошибка при предсказании"
            )
            return "unknown"