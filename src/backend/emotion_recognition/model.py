import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import threading

class EmotionRecognitionModel:
    def __init__(self, model_path: str = 'models/emotion_recognition'):
        """
        Инициализация модели распознавания эмоций
        
        Args:
            model_path (str): Путь для сохранения модели
        """
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.emotions = ['anger', 'joy', 'sadness']
        self.model = None
        self.training_lock = threading.Lock()
        self._create_model()
        self.load()  # Попытка загрузить существующую модель
        
    def _create_model(self):
        """Создание архитектуры модели CNN"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, None, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> Dict:
        """
        Обучение модели
        
        Args:
            X (np.ndarray): Спектрограммы
            y (np.ndarray): Метки классов
            epochs (int): Количество эпох
            
        Returns:
            Dict: История обучения
        """
        if self.model is None:
            self._create_model()
            
        with self.training_lock:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                validation_split=0.2
            )
            return history.history
            
    def predict(self, X: np.ndarray) -> Optional[str]:
        """
        Предсказание эмоции
        
        Args:
            X (np.ndarray): Спектрограммы
            
        Returns:
            Optional[str]: Предсказанная эмоция или None в случае ошибки
        """
        if self.model is None:
            return None
            
        try:
            predictions = self.model.predict(X)
            # Усредняем предсказания по всем спектрограммам
            avg_prediction = np.mean(predictions, axis=0)
            predicted_idx = np.argmax(avg_prediction)
            return self.emotions[predicted_idx]
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            return None
        
    def save(self):
        """Сохранение модели"""
        if self.model is not None:
            self.model.save(self.model_path / 'model.h5')
        
    def load(self) -> bool:
        """
        Загрузка модели
        
        Returns:
            bool: True если модель успешно загружена, False если нет
        """
        try:
            model_file = self.model_path / 'model.h5'
            if model_file.exists():
                self.model = tf.keras.models.load_model(model_file)
                return True
            return False
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            self._create_model()
            return False
        
    def reset(self):
        """Сброс модели"""
        self._create_model() 