import numpy as np
import librosa
import librosa.effects
from io import BytesIO
from backend.api.error_logger import error_logger
from backend.config import SAMPLE_RATE

def load_audio(audio_bytes):
    """
    Загрузка аудио из байтов с использованием librosa
    """
    try:
        audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, res_type='kaiser_fast')
        return audio_data, sr
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_processing",
            "audio_loading",
            "Ошибка при декодировании аудиофайла"
        )
        raise ValueError(f"Не удалось декодировать аудиофайл: {str(e)}")

def normalize_audio(audio_data, audio_bytes=None):
    """
    Нормализация аудиоданных
    """
    try:
        audio_data = librosa.util.normalize(audio_data)
        
        # Проверка результата нормализации
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            error_logger.log_error(
                "Обнаружены недопустимые значения после нормализации",
                "audio_processing",
                "normalization"
            )
            # Если были предоставлены исходные байты, пробуем загрузить заново
            if audio_bytes is not None:
                audio_data, _ = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
            
            audio_data = np.nan_to_num(audio_data)
        
        return audio_data
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_processing",
            "normalization",
            "Ошибка при нормализации аудио"
        )
        raise

def remove_silence(audio_data, audio_bytes=None, sr=SAMPLE_RATE):
    """
    Удаление тишины из аудиоданных
    """
    try:
        audio_data = librosa.effects.trim(
            audio_data, 
            top_db=20,
            frame_length=512,
            hop_length=128
        )[0]
        
        # Проверка результата удаления тишины
        if np.isnan(audio_data).any() or np.isinf(audio_data).any() or len(audio_data) < sr * 0.1:
            error_logger.log_error(
                f"Проблема после удаления тишины: длина={len(audio_data)}",
                "audio_processing",
                "silence_removal"
            )
            # Если были предоставлены исходные байты, пробуем загрузить заново
            if audio_bytes is not None:
                audio_data, _ = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
                audio_data = librosa.util.normalize(audio_data)
        
        return audio_data
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_processing",
            "silence_removal",
            "Ошибка при удалении тишины"
        )
        raise 