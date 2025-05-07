from typing import Dict, Tuple, List, Optional, Union, Any
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from src.backend.loggers.error_logger import error_logger
from src.backend.loggers.info_logger import info_logger

def calculate_metrics(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray],
    y_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: int = 0,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Рассчитывает метрики классификации: accuracy, precision, recall, auc-roc.
    
    Args:
        y_true: Истинные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_pred: Предсказанные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_prob: Вероятности принадлежности к классам (опционально, для расчёта AUC-ROC)
        num_classes: Количество классов
        average: Тип усреднения для precision и recall ('micro', 'macro', 'weighted')
        
    Returns:
        Словарь с рассчитанными метриками
    """
    try:
        # Конвертируем тензоры PyTorch в массивы NumPy, если необходимо
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        metrics = {}
        
        acc = accuracy_score(y_true, y_pred)
        metrics['accuracy'] = acc
            
        # Рассчитываем precision и recall
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['precision'] = prec
        metrics['recall'] = rec
        
        # Рассчитываем AUC-ROC (если доступны вероятности)
        if y_prob is not None:
            if num_classes > 2:
                # Для многоклассовой классификации нужен one-hot encoding для истинных меток
                encoder = OneHotEncoder(sparse_output=False)
                y_true_onehot = encoder.fit_transform(y_true.reshape(-1, 1))
                auc_roc = roc_auc_score(y_true_onehot, y_prob, average=average, multi_class='ovr')
                metrics['auc_roc'] = auc_roc
            else:
                # Для бинарной классификации
                if y_prob.shape[1] == 2:  # Если у нас есть вероятности для обоих классов
                    y_prob = y_prob[:, 1]  # Берём вероятность положительного класса
                auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics['auc_roc'] = auc_roc
        
        return metrics
    except Exception as e:
        error_logger.log_exception(
            e, 
            "metrics_calculation", 
            "calculate_metrics"
        )
        # Возвращаем базовый словарь с нулевыми метриками
        return {}

def log_metrics(metrics: Dict[str, float], epoch: int, num_epochs: int, process_type: str):
    try:
        # Логирование процесса обучения или валидации с расширенными метриками
        log_message = (
            f"Эпоха {epoch+1}/{num_epochs} - "
            f"{process_type}_loss: {metrics.get('loss', 0.0):.4f} - "
            f"{process_type}_acc: {metrics.get('accuracy', 0.0):.4f} - "
            f"{process_type}_prec: {metrics.get('precision', 0.0):.4f} - "
            f"{process_type}_rec: {metrics.get('recall', 0.0):.4f} - "
        )
                
        # Добавляем AUC-ROC, если он был рассчитан
        if 'auc_roc' in metrics:
            log_message += f"{process_type}_auc: {metrics['auc_roc']:.4f} - "
        
        info_logger.info(log_message)
    except Exception as e:
        error_logger.log_exception(e, module="metrics_calculation", context="log_metrics")