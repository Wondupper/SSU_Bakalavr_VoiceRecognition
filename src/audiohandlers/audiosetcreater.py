import librosa
from filter import Filter
from spliter import Spliter

class AudiosetCreater:
    
    def create(self, audio_path):
        filter = Filter()
        spliter = Spliter()
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        cleaned_audio = filter.filter(y, sr)
        
        audioset = spliter.split(cleaned_audio,sr)
        
        return audioset, sr