import os
import numpy as np
import librosa
from io import BytesIO
from backend.api.error_logger import error_logger

# Константы
AUDIO_FRAGMENT_LENGTH = 3  # длина фрагмента в секундах
SAMPLE_RATE = 16000  # частота дискретизации

def process_audio(audio_file):
    """
    Основная функция обработки аудиофайла
    Принимает объект FileStorage и возвращает обработанные аудиоданные
    """
    try:
        # Проверка объекта файла
        if not audio_file:
            raise ValueError("Аудиофайл не предоставлен")
        
        # Сохраняем позицию файла, чтобы вернуть к ней в конце
        original_position = audio_file.tell()
        
        # Чтение аудиофайла из объекта FileStorage
        audio_bytes = audio_file.read()
        
        # Возвращаем файл в исходное положение
        audio_file.seek(original_position)
        
        # Проверка размера файла
        if len(audio_bytes) == 0:
            raise ValueError("Пустой аудиофайл")
        
        # Проверка формата файла
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        if file_extension not in ['.wav', '.mp3', '.ogg', '.flac']:
            raise ValueError(f"Неподдерживаемый формат аудиофайла: {file_extension}. Поддерживаемые форматы: WAV, MP3, OGG, FLAC")
        
        try:
            # Загрузка аудио в память
            audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
        except Exception as load_error:
            raise ValueError(f"Не удалось декодировать аудиофайл: {str(load_error)}")
        
        # Проверка на пустое аудио
        if len(audio_data) == 0:
            raise ValueError("Аудиофайл пуст или поврежден")
        
        # Удаление шума и тишины
        audio_data = remove_noise(audio_data)
        audio_data = remove_silence(audio_data, sr)
        
        # Разделение на фрагменты
        audio_fragments = split_audio(audio_data, sr)
        
        # Проверка, что получены хотя бы какие-то фрагменты
        if not audio_fragments:
            raise ValueError("Не удалось извлечь аудиофрагменты подходящей длины")
        
        return audio_fragments
        
    except Exception as e:
        # Логирование ошибки
        error_message = f"Ошибка обработки аудио: {str(e)}"
        error_logger.log_error(error_message, "audio", "audio_processor")
        raise ValueError(error_message)

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
    # Проверка на пустые данные
    if len(audio_data) == 0:
        return audio_data
        
    # Детектирование тишины с помощью энергетического порога
    try:
        intervals = librosa.effects.split(audio_data, top_db=20)
    except Exception as e:
        error_logger.log_error(f"Ошибка при разделении аудио: {str(e)}", "audio", "remove_silence")
        return audio_data
    
    # Если интервалы не обнаружены, возвращаем исходный сигнал
    if len(intervals) == 0:
        return audio_data
    
    # Объединение ненулевых интервалов
    non_silent_audio = []
    for interval in intervals:
        non_silent_audio.extend(audio_data[interval[0]:interval[1]])
    
    # Проверка наличия аудио после удаления тишины
    if len(non_silent_audio) == 0:
        return audio_data  # Если после удаления тишины ничего не осталось, вернем исходный сигнал
    
    return np.array(non_silent_audio)

def split_audio(audio_data, sr):
    """
    Разделение аудио на фрагменты фиксированной длины с перекрытием
    """
    # Проверка на пустые или некорректные данные
    if audio_data is None or len(audio_data) == 0:
        return []
    
    # Расчет размера фрагмента в отсчетах
    fragment_size = int(AUDIO_FRAGMENT_LENGTH * sr)
    
    # Если аудио слишком короткое, дополняем тишиной
    if len(audio_data) < fragment_size:
        if len(audio_data) >= 1024:  # Минимальный размер для обработки
            padding = np.zeros(fragment_size - len(audio_data))
            padded_audio = np.concatenate([audio_data, padding])
            return [padded_audio]
        return []
    
    # Добавляем перекрытие для лучшего захвата признаков
    hop_size = fragment_size // 2  # 50% перекрытие
    
    # Получение количества фрагментов с учетом перекрытия
    num_fragments = max(1, (len(audio_data) - fragment_size) // hop_size + 1)
    
    # Разделение на фрагменты с перекрытием
    fragments = []
    for i in range(num_fragments):
        start = i * hop_size
        end = min(start + fragment_size, len(audio_data))  # Защита от выхода за границы
        
        # Проверка, что фрагмент достаточной длины
        if end - start >= fragment_size * 0.75:  # Не менее 75% от нужной длины
            fragment = audio_data[start:end]
            
            # Если фрагмент короче fragment_size, дополняем его нулями
            if len(fragment) < fragment_size:
                padding = np.zeros(fragment_size - len(fragment))
                fragment = np.concatenate([fragment, padding])
            
            fragments.append(fragment)
    
    # Проверка на наличие результатов
    if not fragments and len(audio_data) >= fragment_size * 0.5:
        # Если фрагменты не выделены, но аудио не совсем короткое
        # Берем первый фрагмент и дополняем его при необходимости
        fragment = audio_data[:min(fragment_size, len(audio_data))]
        if len(fragment) < fragment_size:
            padding = np.zeros(fragment_size - len(fragment))
            fragment = np.concatenate([fragment, padding])
        fragments.append(fragment)
    
    return fragments
