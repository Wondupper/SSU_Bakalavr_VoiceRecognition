import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

class Filter:
    
    def filter(self, y, sr):
        y_denoised = self.remove_noise(y, sr)
        cleaned_audio = self.remove_silence(y_denoised, sr)
        return cleaned_audio


    # Шумоподавление
    def remove_noise(self, y, sr):
        noise_sample = y[0:int(0.5 * sr)]  # первые 0.5 секунды как пример шума
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
        return y_denoised


    # Функция для удаления тишины
    def remove_silence(self, y, sr):
        
        # Определяет, что считать тишиной. Чем выше значение, тем больше участков будет считаться громкими.
        threshold_db=-35
        
        # Уменьшение frame_length и hop_length увеличивает точность, но замедляет обработку.
        
        frame_length=2048 # Размер окна анализа
        
        hop_length=512 # Шаг
        
        # Расчёт RMS-громкости (среднеквадратичное значение)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Определение неслышных участков
        non_silent_indices = np.where(rms_db > threshold_db)[0]
        if len(non_silent_indices) == 0:
            return np.array([])  # Если нет неслышных участков, вернуть пустое

        # Конвертация индексов фреймов в сэмплы
        frames = librosa.frames_to_samples(non_silent_indices, hop_length=hop_length)
        cleaned_audio = np.concatenate([y[frames[i]:frames[i+1]] for i in range(len(frames)-1)])
        
        return cleaned_audio
