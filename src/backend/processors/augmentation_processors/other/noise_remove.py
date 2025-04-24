import noisereduce as nr
import numpy as np
import random
import librosa
from backend.api.error_logger import error_logger
from backend.config import SAMPLE_RATE, AUGMENTATION_PROCESSOR

# Используем константы из конфигурационного файла
MIN_SNR_DB = AUGMENTATION_PROCESSOR['MIN_SNR_DB']
MAX_SNR_DB = AUGMENTATION_PROCESSOR['MAX_SNR_DB']

def remove_noise(audio_data):
    """
    Удаление шума с использованием библиотеки noisereduce
    """
    # Проверка на пустые данные
    if audio_data is None or len(audio_data) == 0:
        return audio_data
        
    # Для очень коротких аудио пропускаем шумоподавление
    if len(audio_data) < 2048:
        return audio_data
        
    try:
        return nr.reduce_noise(
            y=audio_data, 
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=0.75
        )
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "augmentation",
            "remove_noise",
            "Ошибка при удалении шума"
        )
        # Возвращаем исходные данные в случае ошибки
        return audio_data

def add_noise_with_controlled_snr(audio):
    """
    Добавляет шум к аудиосигналу с контролируемым отношением сигнал/шум.
    
    Args:
        audio: numpy массив с аудиоданными
        
    Returns:
        np.ndarray: аудиоданные с добавленным шумом
    """
    try:
        # Выбираем случайное значение SNR из диапазона
        snr_db = random.uniform(MIN_SNR_DB, MAX_SNR_DB)
        return add_noise_with_snr(audio, snr_db)
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "augmentation",
            "add_noise_with_controlled_snr",
            f"Ошибка при добавлении шума: {str(e)}"
        )
        return audio

def add_noise_with_snr(audio, snr_db):
    """
    Добавляет белый шум к аудиосигналу с заданным отношением сигнал/шум в дБ.
    
    Args:
        audio: numpy массив с аудиоданными
        snr_db: отношение сигнал/шум в дБ
        
    Returns:
        np.ndarray: аудиоданные с добавленным шумом
    """
    try:
        # Преобразуем SNR из дБ в линейную шкалу
        snr_linear = 10 ** (snr_db / 10)
        
        # Расчет среднеквадратичной мощности сигнала
        audio_power = np.mean(audio ** 2)
        audio_rms = np.sqrt(audio_power)
        
        # Расчет необходимой мощности шума
        noise_rms = audio_rms / snr_linear
        
        # Генерация белого шума
        noise = np.random.normal(0, noise_rms, len(audio))
        
        # Добавление шума к сигналу
        noisy_audio = audio + noise
        
        # Нормализация с использованием библиотечной функции
        return librosa.util.normalize(noisy_audio)
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "augmentation",
            "add_noise_with_snr",
            f"Ошибка при добавлении шума с SNR: {str(e)}"
        )
        return audio 