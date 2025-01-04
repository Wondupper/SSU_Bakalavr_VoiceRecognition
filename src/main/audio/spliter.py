import math
import librosa
import soundfile as sf

class Spliter:

    def split(self, y, sr):
        audiolist = list()
        
        # Задание длины одного сегмента (в секундах)
        segment_length = 3  # Длина сегмента в секундах
        
        samples_per_segment = segment_length * sr  # Количество сэмплов на сегмент

        # Определение количества сегментов
        num_segments = math.trunc(len(y) / samples_per_segment)

        # Разделение аудиофайла и сохранение сегментов
        for i in range(num_segments):
            start_sample = int(i * samples_per_segment)
            end_sample = int((i + 1) * samples_per_segment)
            segment = y[start_sample:end_sample]
            audiolist.append(segment)

        return audiolist
