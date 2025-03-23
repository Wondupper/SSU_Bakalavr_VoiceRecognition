import numpy as np
from typing import List
from .audio_processor import AudioProcessor

class SpectrogramProcessor:
    def __init__(self, n_mels: int = 128, sample_rate: int = 16000):
        """
        Инициализация процессора спектрограмм
        
        Args:
            n_mels (int): Количество мел-фильтров
            sample_rate (int): Частота дискретизации
        """
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.n_mels = n_mels
        
    def process_audio_fragments(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Обработка списка аудио фрагментов
        
        Args:
            fragments (List[np.ndarray]): Список аудио фрагментов
            
        Returns:
            List[np.ndarray]: Список мел-спектрограмм
        """
        return [self.audio_processor.create_mel_spectrogram(fragment, self.n_mels) for fragment in fragments] 