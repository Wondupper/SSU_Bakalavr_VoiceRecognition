import librosa
from main.audio.filter import Filter
from main.audio.spliter import Spliter
import soundfile as sf

class Processor:
    
    def process(self, audio_path):
        filter = Filter()
        spliter = Spliter()
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        cleaned_audio = filter.filter(y, sr)
        
        audioset = spliter.split(cleaned_audio,sr)
        
        return audioset, sr
    