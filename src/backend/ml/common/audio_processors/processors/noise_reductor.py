import torch
import torchaudio
from src.backend.loggers.error_logger import error_logger

def apply_noise_reduction(waveform: torch.Tensor) -> torch.Tensor:
    """
    Применяет шумоподавление к аудиоформе
    
    Args:
        waveform: Тензор аудиоформы
    
    Returns:
        Аудиоформа с подавленным шумом
    """
    try:
        # 1. Вычисляем спектрограмму с окном Хэмминга
        window_fn = torch.hamming_window
        spec: torch.Tensor = torchaudio.transforms.Spectrogram(
            n_fft=1024, 
            hop_length=512,
            window_fn=window_fn,
            power=2
        )(waveform)
        
        # 2. Оценка шума из нескольких самых тихих фреймов
        frame_energies = torch.sum(spec, dim=1)
        num_noise_frames = min(20, spec.size(2) // 4)
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
            n_iter=32
        )
        enhanced_waveform: torch.Tensor = griffin_lim(enhanced_spec)
        
        return enhanced_waveform
    
    except Exception as e:
        error_logger.log_exception(
            e,
            "noise_reductor",
            "apply_noise_reduction",
            "Ошибка при шумоподвалении"
        )
        return None