from backend.api.error_logger import error_logger
from backend.config import  AUDIO_FRAGMENT_LENGTH
from .other.validator import validate_audio_file, validate_audio_bytes, validate_audio_data
from .other.normalizer import load_audio, normalize_audio, remove_silence
from .other.splitter import split_audio_into_fixed_length_segments, create_default_fragment

def process_audio(audio_file):
    """
    Основная функция обработки аудиофайла
    """
    try:
        # Проверка объекта файла
        validate_audio_file(audio_file)
        
        # Чтение аудиофайла
        original_position = audio_file.tell()
        audio_bytes = audio_file.read()
        audio_file.seek(original_position)
        
        # Проверка размера файла
        validate_audio_bytes(audio_bytes)
        
        try:
            # Загрузка аудио с использованием librosa
            audio_data, sr = load_audio(audio_bytes)
        except Exception as e:
            error_logger.log_exception(
                e,
                "audio_processing",
                "audio_loading",
                "Ошибка при декодировании аудиофайла"
            )
            raise ValueError(f"Не удалось декодировать аудиофайл: {str(e)}")
        
        # Проверка на пустое аудио
        audio_data = validate_audio_data(audio_data, sr)
        
        try:
            # Нормализация аудио
            audio_data = normalize_audio(audio_data, audio_bytes)
            
            # Удаление тишины
            audio_data = remove_silence(audio_data, audio_bytes, sr)
        
        except Exception as e:
            error_logger.log_exception(
                e,
                "audio_processing",
                "processing",
                "Ошибка при обработке аудио"
            )
            # Используем прямую загрузку без обработки при любой ошибке
            audio_data, sr = load_audio(audio_bytes)
        
        # Проверка размера аудио перед разделением
        min_length = int(AUDIO_FRAGMENT_LENGTH * sr * 0.25)
        validate_audio_data(audio_data, sr, min_length)
        
        # Разделение на фрагменты используя оптимизированный метод
        audio_fragments = split_audio_into_fixed_length_segments(audio_data, sr)
        
        if not audio_fragments:
            error_logger.log_error(
                "Не удалось получить фрагменты, создаем один фрагмент вручную",
                "audio_processing",
                "audio_splitting"
            )
            # Создаем один фрагмент фиксированной длины
            audio_fragments = create_default_fragment(audio_data, sr)
        
        return audio_fragments
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "audio_processing",
            "process_audio",
            "Ошибка обработки аудио"
        )
        raise
