from typing import Dict, Tuple, List, Optional, Union, Any
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from backend.loggers.error_logger import error_logger
from backend.loggers.info_logger import info_logger

def calculate_metrics(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray],
    y_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: int = 0,
    average: str = 'macro',
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Рассчет метрик классификации: accuracy, precision, recall, auc-roc.
    
    Args:
        y_true: Истинные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_pred: Предсказанные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_prob: Вероятности принадлежности к классам (опционально, для расчёта AUC-ROC)
        num_classes: Количество классов
        average: Тип усреднения для precision и recall ('micro', 'macro', 'weighted')
        class_names: Список названий классов (опционально)
        
    Returns:
        Словарь с рассчитанными метриками, включая метрики для каждого класса
    """
    try:
        # Конвертируем тензоры PyTorch в массивы NumPy, если необходимо
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        # Получаем макро-метрики (глобальные)
        metrics = calculate_macro_metrics(y_true, y_pred, y_prob, num_classes, average)
        
        # Получаем микро-метрики (по каждому классу)
        if num_classes > 0:
            per_class_metrics = calculate_micro_metrics(y_true, y_pred, y_prob, num_classes, class_names)
            if per_class_metrics:
                metrics['per_class'] = per_class_metrics
        
        return metrics
    except Exception as e:
        error_logger.log_exception(
            e, 
            "metrics_calculation", 
            "calculate_metrics"
        )
        # Возвращаем базовый словарь с нулевыми метриками
        return {}
    
    
def log_metrics(metrics: Dict[str, Union[float, Dict[str, float]]], epoch: int, num_epochs: int, process_type: str):
    """
    Логирование всех метрик (глобальные и по классам).
    
    Args:
        metrics: Словарь с метриками
        epoch: Номер текущей эпохи
        num_epochs: Общее количество эпох
        process_type: Тип процесса ('train' или 'val')
    """
    try:
        # Логирование глобальных метрик
        log_macro_metrics(metrics, epoch, num_epochs, process_type)
        
        # Логирование метрик по классам
        log_micro_metrics(metrics, epoch, num_epochs, process_type)
    except Exception as e:
        error_logger.log_exception(e, module="metrics_calculation", context="log_metrics")
    

def calculate_macro_metrics(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray],
    y_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: int = 0,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Рассчет глобальных метрик классификации: accuracy, precision, recall, auc-roc.
    
    Args:
        y_true: Истинные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_pred: Предсказанные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_prob: Вероятности принадлежности к классам (опционально, для расчёта AUC-ROC)
        num_classes: Количество классов
        average: Тип усреднения для precision и recall ('micro', 'macro', 'weighted')
        
    Returns:
        Словарь с рассчитанными глобальными метриками
    """
    try:
        metrics = {}
        
        # Рассчитываем глобальные метрики
        acc = accuracy_score(y_true, y_pred)
        metrics['accuracy'] = acc
            
        # Рассчитываем глобальные precision и recall
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['precision'] = prec
        metrics['recall'] = rec
        
        # Рассчитываем глобальный AUC-ROC (если доступны вероятности)
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
            "calculate_macro_metrics"
        )
        return {}

def calculate_micro_metrics(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray],
    y_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: int = 0,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Рассчет метрик классификации для каждого класса отдельно.
    
    Args:
        y_true: Истинные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_pred: Предсказанные метки классов (может быть тензором PyTorch или массивом NumPy)
        y_prob: Вероятности принадлежности к классам (опционально, для расчёта AUC-ROC)
        num_classes: Количество классов
        class_names: Список названий классов (опционально)
        
    Returns:
        Словарь с рассчитанными метриками для каждого класса
    """
    try:
        per_class_metrics = {}
        
        if num_classes <= 0:
            return per_class_metrics
        
        # Вычисляем precision и recall для каждого класса
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Определяем названия классов
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        # Формируем метрики для каждого класса
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            class_metrics = {
                'precision': per_class_precision[i],
                'recall': per_class_recall[i]
            }
            
            # Для AUC-ROC по классам (если доступны вероятности)
            if y_prob is not None:
                # Рассчитываем AUC-ROC для текущего класса
                class_true = (y_true == i).astype(int)
                class_prob = y_prob[:, i] if y_prob.ndim > 1 else y_prob
                try:
                    class_auc = roc_auc_score(class_true, class_prob)
                    class_metrics['auc_roc'] = class_auc
                except:
                    pass  # Если для класса недостаточно данных, пропускаем AUC
            
            per_class_metrics[class_name] = class_metrics
        
        return per_class_metrics
    except Exception as e:
        error_logger.log_exception(
            e, 
            "metrics_calculation", 
            "calculate_micro_metrics"
        )
        return {}


def log_macro_metrics(metrics: Dict[str, Union[float, Dict[str, float]]], epoch: int, num_epochs: int, process_type: str):
    """
    Логирование глобальных метрик обучения или валидации.
    
    Args:
        metrics: Словарь с метриками
        epoch: Номер текущей эпохи
        num_epochs: Общее количество эпох
        process_type: Тип процесса ('train' или 'val')
    """
    try:
        # Логирование глобальных метрик процесса обучения или валидации
        log_message = (
            f"Эпоха {epoch+1}/{num_epochs} - "
            f"{process_type}_loss: {metrics.get('loss', 0.0):.4f} - "
            f"{process_type}_acc: {metrics.get('accuracy', 0.0):.4f} - "
            f"{process_type}_prec: {metrics.get('precision', 0.0):.4f} - "
            f"{process_type}_rec: {metrics.get('recall', 0.0):.4f}"
        )
                
        # Добавляем AUC-ROC, если он был рассчитан
        if 'auc_roc' in metrics:
            log_message += f" - {process_type}_auc: {metrics['auc_roc']:.4f}"
        
        info_logger.info(log_message)
    except Exception as e:
        error_logger.log_exception(e, module="metrics_calculation", context="log_macro_metrics")

def log_micro_metrics(metrics: Dict[str, Union[float, Dict[str, float]]], epoch: int, num_epochs: int, process_type: str):
    """
    Логирование метрик для каждого класса.
    
    Args:
        metrics: Словарь с метриками, включая 'per_class'
        epoch: Номер текущей эпохи
        num_epochs: Общее количество эпох
        process_type: Тип процесса ('train' или 'val')
    """
    try:
        # Логирование метрик для каждого класса, если они доступны
        if 'per_class' in metrics and metrics['per_class']:
            for class_name, class_metrics in metrics['per_class'].items():
                class_log = (
                    f"Эпоха {epoch+1}/{num_epochs} - {process_type} класс {class_name} - "
                    f"prec: {class_metrics.get('precision', 0.0):.4f} - "
                    f"rec: {class_metrics.get('recall', 0.0):.4f}"
                )
                
                if 'auc_roc' in class_metrics:
                    class_log += f" - auc: {class_metrics['auc_roc']:.4f}"
                
                info_logger.info(class_log)
    except Exception as e:
        error_logger.log_exception(e, module="metrics_calculation", context="log_micro_metrics")
