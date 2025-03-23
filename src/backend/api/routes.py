from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import shutil
from pathlib import Path
from ..processors import AudioProcessor, SpectrogramProcessor, AugmentationProcessor, DatasetCreator
from ..voice_identification import VoiceIdentificationModel
from ..emotion_recognition import EmotionRecognitionModel
from ..config import PathConfig
from .validators import validate_training_data, validate_identification_data
import random

router = APIRouter()

# Инициализация моделей
voice_model = VoiceIdentificationModel()
emotion_model = EmotionRecognitionModel()

# Инициализация процессоров
audio_processor = AudioProcessor()
spectrogram_processor = SpectrogramProcessor()
augmentation_processor = AugmentationProcessor()
dataset_creator = DatasetCreator()

@router.post("/train")
async def train_model(
    files: List[UploadFile] = File(...),
    user_name: str = Form(...)
):
    """
    Эндпоинт для обучения моделей на новых данных
    """
    # Валидация входных данных
    await validate_training_data(files, user_name)
    
    try:
        # Сохранение файлов во временную директорию
        saved_files = []
        for file in files:
            file_path = PathConfig.TEMP_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
        try:
            # Обработка аудио файлов
            all_spectrograms = []
            for file_path in saved_files:
                spectrograms, _ = audio_processor.process_audio_file(str(file_path))
                all_spectrograms.extend(spectrograms)
                
            # Аугментация спектрограмм
            augmented_spectrograms = augmentation_processor.process_spectrograms(all_spectrograms)
            
            # Создание датасетов и обучение моделей
            voice_dataset = dataset_creator.create_voice_identification_dataset(
                augmented_spectrograms,
                user_name
            )
            voice_history = voice_model.train(voice_dataset['spectrograms'], voice_dataset['labels'])
            
            emotion_dataset = dataset_creator.create_emotion_recognition_dataset(
                augmented_spectrograms,
                ["neutral"] * len(augmented_spectrograms)  # Временная метка
            )
            emotion_history = emotion_model.train(emotion_dataset['spectrograms'], emotion_dataset['labels'])
            
            # Сохранение обученных моделей
            voice_model.save()
            emotion_model.save()
            
            return {
                "message": "Обучение успешно завершено",
                "voice_accuracy": voice_history.get('accuracy', [])[-1] if voice_history.get('accuracy') else None,
                "emotion_accuracy": emotion_history.get('accuracy', [])[-1] if emotion_history.get('accuracy') else None
            }
            
        finally:
            # Очистка временных файлов
            for file_path in saved_files:
                file_path.unlink()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/identify")
async def identify_user(
    file: UploadFile = File(...),
    emotion: str = Form(...)
):
    """
    Эндпоинт для идентификации пользователя и проверки эмоции
    """
    # Валидация входных данных
    await validate_identification_data(file)
    
    try:
        # Сохранение файла во временную директорию
        file_path = PathConfig.TEMP_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            # Обработка аудио файла
            spectrograms, _ = audio_processor.process_audio_file(str(file_path))
            
            # Получение предсказаний
            voice_match = voice_model.predict(spectrograms)
            predicted_emotion = emotion_model.predict(spectrograms)
            
            if predicted_emotion is None:
                raise HTTPException(
                    status_code=500,
                    detail="Ошибка при распознавании эмоции"
                )
            
            return {
                "voice_match": voice_match,
                "emotion_match": predicted_emotion == emotion,
                "predicted_emotion": predicted_emotion
            }
            
        finally:
            # Очистка временного файла
            file_path.unlink()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/emotions")
async def get_random_emotion():
    """
    Эндпоинт для получения случайной эмоции
    """
    emotions = emotion_model.emotions
    return {"emotion": random.choice(emotions)} 