import numpy as np
import librosa
from backend.api.error_logger import error_logger

def augment_audio(audio_fragments):
    """
    Аугментация аудиофрагментов в соответствии с требованиями
    Возвращает расширенный набор аудиофрагментов
    """
    augmented_fragments = []
    
    # Шаг 1: Добавляем исходные фрагменты
    augmented_fragments.extend(audio_fragments)
    
    # Шаг 2: Удаление шума (группа A)
    denoised_fragments = [remove_noise(fragment) for fragment in audio_fragments]
    augmented_fragments.extend(denoised_fragments)
    
    # Создаем копии всех фрагментов для последующих преобразований
    step1_fragments = augmented_fragments.copy()
    
    # Шаг 3: Ускорение записи (группа B)
    for speed in [1.1, 1.2, 1.3, 1.4, 1.5]:
        speed_fragments = [change_speed(fragment, speed) for fragment in step1_fragments]
        augmented_fragments.extend(speed_fragments)
    
    # Шаг 4: Замедление записи (группа C)
    for speed in [0.9, 0.8, 0.7]:
        speed_fragments = [change_speed(fragment, speed) for fragment in step1_fragments]
        augmented_fragments.extend(speed_fragments)
    
    # Создаем копии всех фрагментов для последующих преобразований
    step4_fragments = augmented_fragments.copy()
    
    # Шаг 5: Изменение тональности (группа D)
    for pitch_shift in [1, 2]:
        pitch_fragments = [change_pitch(fragment, pitch_shift) for fragment in step4_fragments]
        augmented_fragments.extend(pitch_fragments)
    
    # Шаг 6: Изменение тональности (группа E)
    for pitch_shift in [-1, -2]:
        pitch_fragments = [change_pitch(fragment, pitch_shift) for fragment in step1_fragments]
        augmented_fragments.extend(pitch_fragments)
    
    return augmented_fragments

def remove_noise(audio_data):
    """
    Удаление шума из аудиоданных (группа A)
    """
    # Проверка на пустые данные
    if len(audio_data) == 0:
        return audio_data
    
    # Расчет спектрограммы
    stft = librosa.stft(audio_data)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Оценка шумового порога (предполагаем, что первые 100 мс это шум)
    # Обеспечиваем, что noise_idx >= 1, чтобы избежать проблем с пустыми аудио
    noise_idx = max(1, int(100 * 16000 / 1000 / (2048 // 4)))
    
    # Если аудио короче noise_idx кадров, используем первую треть
    if mag.shape[1] <= noise_idx:
        noise_idx = max(1, mag.shape[1] // 3)
    
    noise_profile = np.mean(mag[:, :noise_idx], axis=1, keepdims=True)
    
    # Спектральное вычитание
    mag = np.maximum(mag - noise_profile, 0)
    
    # Обратное преобразование
    stft_denoised = mag * np.exp(1j * phase)
    audio_denoised = librosa.istft(stft_denoised)
    
    return audio_denoised

def change_speed(audio_data, speed_factor):
    """
    Изменение скорости аудиоданных (группы B и C)
    
    Args:
        audio_data: аудиоданные для изменения
        speed_factor: коэффициент изменения скорости
    
    Returns:
        Аудиоданные с измененной скоростью
    """
    try:
        # Проверка на пустые данные
        if len(audio_data) == 0:
            return audio_data
            
        # Проверка на минимальную длину аудио
        if len(audio_data) < 1024:  # Минимальный размер для корректной работы librosa
            return audio_data
            
        # Изменение скорости с сохранением длины
        return librosa.effects.time_stretch(audio_data, rate=speed_factor)
    except Exception as e:
        error_message = f"Ошибка при изменении скорости: {str(e)}"
        error_logger.log_error(error_message, "processing", "augmentation")
        return audio_data

def change_pitch(audio_data, n_steps, sr=16000):
    """
    Изменение высоты тона аудиоданных (группы D и E)
    
    Args:
        audio_data: аудиоданные для изменения
        n_steps: количество полутонов для сдвига
        sr: частота дискретизации (по умолчанию 16000)
    
    Returns:
        Аудиоданные с измененной высотой тона
    """
    try:
        # Проверка на пустые данные
        if len(audio_data) == 0:
            return audio_data
            
        # Проверка на минимальную длину аудио
        if len(audio_data) < 1024:  # Минимальный размер для корректной работы librosa
            return audio_data
            
        # Изменение высоты тона
        return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps)
    except Exception as e:
        error_message = f"Ошибка при изменении высоты тона: {str(e)}"
        error_logger.log_error(error_message, "processing", "augmentation")
        return audio_data
