import os
import torch
import torchaudio
import io
import random
from typing import List, Optional
from werkzeug.datastructures import FileStorage
from src.backend.loggers.error_logger import error_logger
from src.backend.config import SAMPLE_RATE, AUDIO_FRAGMENT_LENGTH, MODELS_PARAMS
from src.backend.ml.common.augmentation import apply_augmentation

def get_features_tensors_from_audio(audio_file: FileStorage) -> List[torch.Tensor]:
    """
    Извлекает признаки из аудиофайла с помощью torchaudio
    
    Args:
        audio_file: Файл аудио (объект FileStorage Flask)
        module_name: Имя модуля для логирования
        
    Returns:
        Список тензоров признаков для каждого фрагмента
    """
    try:
        # Сохраняем содержимое файла в буфер
        audio_buffer: io.BytesIO = io.BytesIO(audio_file.read())
        # Сбрасываем указатель в начало буфера
        audio_buffer.seek(0)
        # Пытаемся загрузить аудио из буфера
        waveform: torch.Tensor
        sample_rate: int
        waveform, sample_rate = torchaudio.load(audio_buffer)
        # Сбрасываем указатель файла на начало для возможного дальнейшего использования
        audio_file.seek(0)
        
        # Делаем ресемплинг до нужной частоты
        if sample_rate != SAMPLE_RATE:
            resampler: torchaudio.transforms.Resample = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Преобразуем в моно, если нужно
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Нормализация
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-6)

        # Улучшенный процесс шумоподавления
        # 1. Вычисляем спектрограмму с окном Хэмминга
        window_fn = torch.hamming_window
        spec: torch.Tensor = torchaudio.transforms.Spectrogram(
            n_fft=1024, 
            hop_length=512,
            window_fn=window_fn,
            power=2
        )(waveform)
        
        # 2. Оценка шума из нескольких самых тихих фреймов (а не только первых)
        frame_energies = torch.sum(spec, dim=1)
        num_noise_frames = min(20, spec.size(2) // 4)  # Анализируем больше фреймов
        _, frame_indices = torch.topk(frame_energies, num_noise_frames, largest=False, dim=1)
        noise_frames = torch.stack([spec[:, :, i] for i in frame_indices[0]], dim=2)
        noise_estimate: torch.Tensor = torch.mean(noise_frames, dim=2, keepdim=True)
        
        # 3. Спектральное вычитание с мягким порогом и сохранением фазы
        enhanced_spec: torch.Tensor = torch.maximum(spec - 2 * noise_estimate, spec * 0.1)
        
        # 4. Обратное преобразование в волновую форму с более плавным синтезом
        griffin_lim: torchaudio.transforms.GriffinLim = torchaudio.transforms.GriffinLim(
            n_fft=1024, 
            hop_length=512,
            window_fn=window_fn,
            power=2,
            n_iter=32  # Увеличиваем число итераций для лучшего качества
        )
        enhanced_waveform: torch.Tensor = griffin_lim(enhanced_spec)
        
        # 5. Детектирование и удаление тишины
        silence_threshold = 0.01
        is_silence = torch.abs(enhanced_waveform) < silence_threshold
        non_silence_indices = torch.where(~is_silence)[1]
        
        if len(non_silence_indices) > 0:
            start_idx = max(0, non_silence_indices[0] - int(0.1 * SAMPLE_RATE))  # Добавляем 100мс до начала
            end_idx = min(enhanced_waveform.size(1), non_silence_indices[-1] + int(0.1 * SAMPLE_RATE))  # Добавляем 100мс после конца
            enhanced_waveform = enhanced_waveform[:, start_idx:end_idx]
            
            # Повторная нормализация после удаления тишины
            enhanced_waveform = enhanced_waveform / (torch.max(torch.abs(enhanced_waveform)) + 1e-6)

        # Применяем аугментацию к очищенной аудиоформе
        augmented_waveforms: List[torch.Tensor] = apply_augmentation(enhanced_waveform)
        
        # Разбиение каждой аугментированной аудиоформы на фрагменты
        features_list: List[torch.Tensor] = []
        
        for aug_waveform in augmented_waveforms:
            # Разбиение на фрагменты с перекрытием для увеличения числа образцов
            fragment_length: int = int(SAMPLE_RATE * AUDIO_FRAGMENT_LENGTH)
            hop_length: int = fragment_length // 2  # 50% перекрытие для увеличения числа фрагментов
            
            # Количество фрагментов с учетом перекрытия
            num_fragments: int = max(1, 1 + (aug_waveform.size(1) - fragment_length) // hop_length)
            
            for i in range(num_fragments):
                start: int = i * hop_length
                end: int = min(start + fragment_length, aug_waveform.size(1))
                
                fragment: torch.Tensor = aug_waveform[:, start:end]
                
                # Если фрагмент слишком короткий, дополняем его нулями
                if end - start < fragment_length:
                    padding: torch.Tensor = torch.zeros(1, fragment_length - (end - start))
                    fragment = torch.cat([fragment, padding], dim=1)
                
                # Извлечение разнообразных признаков
                # 1. MFCC признаки
                mfcc_transform: torchaudio.transforms.MFCC = torchaudio.transforms.MFCC(
                    sample_rate=SAMPLE_RATE,
                    n_mfcc=40,
                    log_mels=True,
                    melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128}
                )
                mfcc: torch.Tensor = mfcc_transform(fragment)
                
                # 2. Дельта и дельта-дельта коэффициенты
                delta: torch.Tensor = torchaudio.functional.compute_deltas(mfcc)
                delta2: torch.Tensor = torchaudio.functional.compute_deltas(delta)
                
                # 3. Спектральные признаки: спектральный центроид и спектральный разброс
                spectrogram = torchaudio.transforms.Spectrogram(
                    n_fft=2048,
                    hop_length=512
                )(fragment)
                
                # Уменьшаем размерность спектральных признаков до соответствия MFCC
                target_time_dim = mfcc.shape[2]
                if spectrogram.shape[2] > target_time_dim:
                    # Уменьшаем количество временных кадров
                    indices = torch.linspace(0, spectrogram.shape[2] - 1, target_time_dim).long()
                    spectrogram = spectrogram[:, :, indices]
                
                # 4. Сжимаем размерность спектрограммы для соответствия MFCC
                # Используем усреднение по частотным диапазонам
                spectral_bands = 40  # Столько же, сколько MFCC коэффициентов
                freq_indices = torch.linspace(0, spectrogram.shape[1] - 1, spectral_bands).long()
                spec_features = spectrogram[:, freq_indices, :]
                
                # 5. Объединяем все признаки
                combined_features: torch.Tensor = torch.cat([mfcc, delta, delta2, spec_features], dim=1)
                
                # Делаем pad или обрезаем до фиксированной длины
                target_length: int = MODELS_PARAMS['FEATURE_TARGET_LENGTH']
                if combined_features.size(2) < target_length:
                    pad: torch.Tensor = torch.zeros(1, combined_features.size(1), target_length - combined_features.size(2))
                    combined_features = torch.cat([combined_features, pad], dim=2)
                else:
                    combined_features = combined_features[:, :, :target_length]
                
                # Добавляем в список признаков
                features_list.append(combined_features.squeeze(0).transpose(0, 1))
        
            
        return features_list
        
    except Exception as e:
        error_logger.log_exception(
            e,
            "features_tensors_extractor",
            "get_features_tensors_from_audio",
            "Ошибка при извлечении признаков"
        )
        return []