import numpy as np
from typing import List, Dict

class DatasetCreator:
    def create_voice_identification_dataset(
        self,
        spectrograms: List[np.ndarray],
        user_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Создание датасета для идентификации голоса
        
        Args:
            spectrograms (List[np.ndarray]): Список спектрограмм
            user_name (str): Имя пользователя
            
        Returns:
            Dict[str, np.ndarray]: Датасет с спектрограммами и метками
        """
        X = np.array(spectrograms)
        y = np.array([1] * len(spectrograms))  # 1 для целевого пользователя
        
        return {
            'spectrograms': X,
            'labels': y,
            'user_name': user_name
        }
        
    def create_emotion_recognition_dataset(
        self,
        spectrograms: List[np.ndarray],
        emotions: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Создание датасета для распознавания эмоций
        
        Args:
            spectrograms (List[np.ndarray]): Список спектрограмм
            emotions (List[str]): Список эмоций
            
        Returns:
            Dict[str, np.ndarray]: Датасет с спектрограммами и метками
        """
        X = np.array(spectrograms)
        y = np.array(emotions)
        
        return {
            'spectrograms': X,
            'labels': y
        }