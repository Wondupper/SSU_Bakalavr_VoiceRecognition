import librosa
import numpy as np
import soundfile as sf
from typing import List, Tuple

class AudioProcessor:
    def __init__(self, fragment_length: int = 3, sample_rate: int = 16000):
        """
        Инициализация процессора аудио
        
        Args:
            fragment_length (int): Длина фрагмента аудио в секундах
            sample_rate (int): Частота дискретизации
        """
        self.fragment_length = fragment_length
        self.sample_rate = sample_rate

    def remove_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Удаление шума из аудио сигнала
        
        Args:
            audio (np.ndarray): Аудио сигнал
            
        Returns:
            np.ndarray: Очищенный аудио сигнал
        """
        # Реализация удаления шума (будет добавлена позже)
        return audio

    def remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Удаление тишины из аудио сигнала
        
        Args:
            audio (np.ndarray): Аудио сигнал
            
        Returns:
            np.ndarray: Аудио сигнал без тишины
        """
        # Реализация удаления тишины (будет добавлена позже)
        return audio

    def split_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Разделение аудио на фрагменты заданной длины
        
        Args:
            audio (np.ndarray): Аудио сигнал
            
        Returns:
            List[np.ndarray]: Список фрагментов аудио
        """
        fragment_length_samples = self.fragment_length * self.sample_rate
        fragments = []
        
        for i in range(0, len(audio), fragment_length_samples):
            fragment = audio[i:i + fragment_length_samples]
            if len(fragment) == fragment_length_samples:
                fragments.append(fragment)
                
        return fragments

    def create_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Создание мел-спектрограммы из аудио
        
        Args:
            audio (np.ndarray): Аудио сигнал
            n_mels (int): Количество мел-фильтров
            
        Returns:
            np.ndarray: Мел-спектрограмма
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def process_audio_file(self, audio_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Полная обработка аудиофайла
        
        Args:
            audio_path (str): Путь к аудио файлу
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Кортеж (спектрограммы, фрагменты)
        """
        # Загрузка и ресемплирование аудио
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Удаление шумов и тишины
        audio = self.remove_noise(audio)
        audio = self.remove_silence(audio)
        
        # Разделение на фрагменты
        segments = self.split_audio(audio)
        
        # Создание мел-спектрограмм
        spectrograms = [self.create_mel_spectrogram(segment) for segment in segments]
        
        return spectrograms, segments 