import numpy as np
import librosa
from backend.api.error_logger import error_logger

def augment_audio(audio_fragments):
    """
    Аугментация аудиофрагментов в соответствии с требованиями
    Возвращает расширенный набор аудиофрагментов
    """
    # Шаг 1: Исходные фрагменты
    result_fragments = []
    result_fragments.extend(audio_fragments)
    
    # Шаг 2: Удаление шума (группа A)
    denoised_fragments = [remove_noise(fragment) for fragment in audio_fragments]
    result_fragments.extend(denoised_fragments)
    
    # Набор после шагов 1-2
    step1_2_fragments = result_fragments.copy()
    
    # Шаг 3: Ускорение записи (группа B)
    group_b_fragments = []
    for speed in [1.1, 1.2, 1.3, 1.4, 1.5]:
        speed_fragments = [change_speed(fragment, speed) for fragment in step1_2_fragments]
        group_b_fragments.extend(speed_fragments)
    
    # Шаг 4: Замедление записи (группа C)
    group_c_fragments = []
    for speed in [0.9, 0.8, 0.7]:
        speed_fragments = [change_speed(fragment, speed) for fragment in step1_2_fragments]
        group_c_fragments.extend(speed_fragments)
    
    # Объединяем результаты шагов 1-4
    result_fragments.extend(group_b_fragments)
    result_fragments.extend(group_c_fragments)
    
    # Набор после шагов 1-4 (для группы D)
    steps_1_4_fragments = result_fragments.copy()
    
    # Шаг 5: Изменение тональности вверх (группа D) - применяется ко всему набору
    group_d_fragments = []
    for pitch_shift in [1, 2]:
        pitch_fragments = [change_pitch(fragment, pitch_shift) for fragment in steps_1_4_fragments]
        group_d_fragments.extend(pitch_fragments)
    
    # Шаг 6: Изменение тональности вниз (группа E) - применяется к набору 1-3
    # Для этого создаем набор 1-3 (исходные + шум + ускорение + замедление)
    steps_1_3_fragments = []
    steps_1_3_fragments.extend(step1_2_fragments)
    steps_1_3_fragments.extend(group_b_fragments)
    steps_1_3_fragments.extend(group_c_fragments)
    
    group_e_fragments = []
    for pitch_shift in [-1, -2]:
        pitch_fragments = [change_pitch(fragment, pitch_shift) for fragment in steps_1_3_fragments]
        group_e_fragments.extend(pitch_fragments)
    
    # Объединяем все результаты
    result_fragments.extend(group_d_fragments)
    result_fragments.extend(group_e_fragments)
    
    return result_fragments

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
