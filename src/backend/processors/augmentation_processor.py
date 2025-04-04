import numpy as np
import librosa

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
    Реализация та же, что и в audio_processor.py
    """
    # Расчет спектрограммы
    stft = librosa.stft(audio_data)
    mag = np.abs(stft)
    phase = np.angle(stft)
    
    # Оценка шумового порога (предполагаем, что первые 100 мс это шум)
    noise_idx = int(100 * 16000 / 1000 / (2048 // 4))
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
    """
    # Изменение скорости с сохранением длины
    return librosa.effects.time_stretch(audio_data, rate=speed_factor)

def change_pitch(audio_data, n_steps):
    """
    Изменение высоты тона аудиоданных (группы D и E)
    """
    # Изменение высоты тона
    return librosa.effects.pitch_shift(audio_data, sr=16000, n_steps=n_steps)
