import os
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO

# Константы
AUDIO_FRAGMENT_LENGTH = 3  # длина фрагмента в секундах
SAMPLE_RATE = 16000  # частота дискретизации

def process_audio(audio_file):
    """
    Основная функция обработки аудиофайла
    Принимает объект FileStorage и возвращает обработанные аудиоданные
    """
    # Чтение аудиофайла из объекта FileStorage
    audio_bytes = audio_file.read()
    
    # Загрузка аудио в память
    audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
    
    # Удаление шума и тишины
    audio_data = remove_noise(audio_data)
    audio_data = remove_silence(audio_data, sr)
    
    # Разделение на фрагменты
    audio_fragments = split_audio(audio_data, sr)
    
    return audio_fragments

def remove_noise(audio_data):
    """
    Удаление шума из аудиоданных
    Простая реализация с использованием спектрального вычитания
    """
    # Расчет спектрограммы
    stft = librosa.stft(audio_data)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Оценка шумового порога (предполагаем, что первые 1000 мс это шум)
    noise_idx = int(1000 * SAMPLE_RATE / 1000 / (2048 // 4))
    noise_profile = np.mean(mag[:, :noise_idx], axis=1, keepdims=True)
    
    # Спектральное вычитание
    mag = np.maximum(mag - noise_profile, 0)
    
    # Обратное преобразование
    stft_denoised = mag * np.exp(1j * phase)
    audio_denoised = librosa.istft(stft_denoised)
    
    return audio_denoised

def remove_silence(audio_data, sr):
    """
    Удаление тишины из аудиоданных
    """
    # Детектирование тишины с помощью энергетического порога
    intervals = librosa.effects.split(audio_data, top_db=20)
    
    # Если интервалы не обнаружены, возвращаем исходный сигнал
    if len(intervals) == 0:
        return audio_data
    
    # Объединение ненулевых интервалов
    non_silent_audio = []
    for interval in intervals:
        non_silent_audio.extend(audio_data[interval[0]:interval[1]])
    
    return np.array(non_silent_audio)

def split_audio(audio_data, sr):
    """
    Разделение аудио на фрагменты фиксированной длины
    """
    # Расчет размера фрагмента в отсчетах
    fragment_size = int(AUDIO_FRAGMENT_LENGTH * sr)
    
    # Получение количества полных фрагментов
    num_fragments = len(audio_data) // fragment_size
    
    # Если аудио короче, чем AUDIO_FRAGMENT_LENGTH, возвращаем пустой список
    if num_fragments == 0:
        return []
    
    # Разделение на фрагменты
    fragments = []
    for i in range(num_fragments):
        start = i * fragment_size
        end = (i + 1) * fragment_size
        fragment = audio_data[start:end]
        fragments.append(fragment)
    
    return fragments
