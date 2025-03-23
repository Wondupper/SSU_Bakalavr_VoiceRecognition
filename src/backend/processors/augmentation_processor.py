import numpy as np
from typing import List
from scipy.ndimage import zoom

class AugmentationProcessor:
    def flip_horizontal(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Отражение спектрограммы по горизонтали
        
        Args:
            spectrogram (np.ndarray): Исходная спектрограмма
            
        Returns:
            np.ndarray: Отраженная спектрограмма
        """
        return np.fliplr(spectrogram)
        
    def flip_vertical(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Отражение спектрограммы по вертикали
        
        Args:
            spectrogram (np.ndarray): Исходная спектрограмма
            
        Returns:
            np.ndarray: Отраженная спектрограмма
        """
        return np.flipud(spectrogram)
        
    def stretch_time(self, spectrogram: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """
        Растяжение спектрограммы по временной оси
        
        Args:
            spectrogram (np.ndarray): Исходная спектрограмма
            factor (float): Коэффициент растяжения
            
        Returns:
            np.ndarray: Растянутая спектрограмма
        """
        stretched = zoom(spectrogram, (1, factor))
        if stretched.shape[1] > spectrogram.shape[1]:
            stretched = stretched[:, :spectrogram.shape[1]]
        return stretched
        
    def stretch_freq(self, spectrogram: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """
        Растяжение спектрограммы по частотной оси
        
        Args:
            spectrogram (np.ndarray): Исходная спектрограмма
            factor (float): Коэффициент растяжения
            
        Returns:
            np.ndarray: Растянутая спектрограмма
        """
        stretched = zoom(spectrogram, (factor, 1))
        if stretched.shape[0] > spectrogram.shape[0]:
            stretched = stretched[:spectrogram.shape[0], :]
        return stretched
        
    def augment_spectrogram(self, spectrogram: np.ndarray) -> List[np.ndarray]:
        """
        Применение всех аугментаций к спектрограмме
        
        Args:
            spectrogram (np.ndarray): Исходная спектрограмма
            
        Returns:
            List[np.ndarray]: Список аугментированных спектрограмм
        """
        return [
            spectrogram,  # оригинальная спектрограмма
            self.flip_horizontal(spectrogram),
            self.flip_vertical(spectrogram),
            self.stretch_time(spectrogram),
            self.stretch_freq(spectrogram)
        ]
        
    def process_spectrograms(self, spectrograms: List[np.ndarray]) -> List[np.ndarray]:
        """
        Обработка списка спектрограмм
        
        Args:
            spectrograms (List[np.ndarray]): Список исходных спектрограмм
            
        Returns:
            List[np.ndarray]: Список аугментированных спектрограмм
        """
        augmented = []
        for spectrogram in spectrograms:
            augmented.extend(self.augment_spectrogram(spectrogram))
        return augmented 