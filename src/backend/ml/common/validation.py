from typing import Dict
import numpy as np
import torch
from src.backend.ml.common.metrics_calculation import calculate_metrics, log_metrics
from src.backend.loggers.error_logger import error_logger

def calculate_batch_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    epoch: int,
    num_epochs: int
) -> Dict[str, float]:
    """
    Рассчитывает метрики на наборе данных с помощью загрузчика данных.
    
    Args:
        model: Модель PyTorch
        data_loader: Загрузчик данных
        device: Устройство для вычислений
        num_classes: Количество классов
        
    Returns:
        Словарь с усреднёнными метриками
    """
    try:
        model.eval()
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Для логирования auc-roc сохраняем вероятности
                probabilities = torch.softmax(outputs, dim=1)
                
                # Получаем предсказанные классы
                _, predicted = torch.max(outputs.data, 1)
                
                all_y_true.extend(targets.cpu().numpy())
                all_y_pred.extend(predicted.cpu().numpy())
                all_y_prob.append(probabilities.cpu().numpy())
                
                # Если предоставлена функция потерь, рассчитываем потери
                if hasattr(model, 'criterion'):
                    loss = model.criterion(outputs, targets)
                    total_loss += loss.item()
                
                num_batches += 1
        
        # Объединяем все batches
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.concatenate(all_y_prob, axis=0) if all_y_prob else None
        
        # Рассчитываем метрики
        metrics = calculate_metrics(all_y_true, all_y_pred, all_y_prob, num_classes=num_classes)
        
        # Добавляем потери, если доступны
        if num_batches > 0 and hasattr(model, 'criterion'):
            metrics['loss'] = total_loss / num_batches
        
        log_metrics(metrics, epoch=epoch, num_epochs=num_epochs, process_type = 'val')
        
        return metrics
    

    except Exception as e:
        error_logger.log_exception(
            e,
            "validation", 
            f"calculate_batch_metrics (epoch {epoch}/{num_epochs})"
        )
        # Возвращаем пустой словарь метрик в случае ошибки
        return {}