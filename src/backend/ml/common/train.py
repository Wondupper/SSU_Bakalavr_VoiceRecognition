from typing import Dict
import numpy as np
import torch
from backend.ml.common.metrics_calculation import calculate_metrics, log_metrics
from backend.loggers.error_logger import error_logger

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> Dict[str, float]:
    """
    Выполняет одну эпоху обучения модели.
    
    Args:
        model: Модель PyTorch
        data_loader: Загрузчик данных
        optimizer: Оптимизатор
        criterion: Функция потерь
        device: Устройство для вычислений
        
    Returns:
        Словарь с метриками обучения
    """
    try:
        model.train()
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()
            
            # Для логирования
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_y_true.extend(targets.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())
            all_y_prob.append(probabilities.detach().cpu().numpy())
            
            total_loss += loss.item() * inputs.size(0)
            num_batches += 1
        
        # Объединяем все батчи
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.concatenate(all_y_prob, axis=0) if all_y_prob else None
        
        # Рассчитываем метрики
        metrics = calculate_metrics(all_y_true, all_y_pred, all_y_prob, num_classes=len(set(all_y_true)))
        
        # Добавляем потери
        if num_batches > 0:
            metrics['loss'] = total_loss / len(data_loader.dataset)
        
        log_metrics(metrics, epoch=epoch, num_epochs=num_epochs, process_type= 'train')
        
        return metrics
    

    except Exception as e:
        error_logger.log_exception(
            e,
            "train", 
            f"train_one_epoch (epoch {epoch}/{num_epochs})"
        )
        # Возвращаем пустой словарь метрик в случае ошибки
        return {}

