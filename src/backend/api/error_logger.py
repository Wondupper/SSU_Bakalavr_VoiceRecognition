import threading
from collections import deque
import time

class ErrorLogger:
    def __init__(self, max_errors=100, retention_days=7):
        self.errors = deque(maxlen=max_errors)
        self._lock = threading.Lock()
        self.retention_days = retention_days
        self._cleanup_old_errors()
    
    def log_error(self, error_message, error_type="system", module=None):
        """
        Записывает ошибку в лог
        
        Args:
            error_message: Текст ошибки
            error_type: Тип ошибки (system, model, audio, etc.)
            module: Модуль, в котором произошла ошибка
        """
        with self._lock:
            timestamp = time.time()
            self.errors.appendleft({
                'timestamp': timestamp,
                'message': error_message,
                'type': error_type,
                'module': module
            })
    
    def get_recent_errors(self, limit=10):
        """
        Возвращает последние ошибки
        
        Args:
            limit: Максимальное количество возвращаемых ошибок
            
        Returns:
            Список последних ошибок
        """
        with self._lock:
            return list(self.errors)[:limit]
    
    def clear_errors(self):
        """Очищает список ошибок"""
        with self._lock:
            self.errors.clear()
    
    def _cleanup_old_errors(self):
        """Удаляет ошибки старше retention_days дней"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (self.retention_days * 24 * 60 * 60)
            
            # Создаем новый список, исключая старые ошибки
            recent_errors = deque(maxlen=self.errors.maxlen)
            for error in self.errors:
                if error['timestamp'] >= cutoff_time:
                    recent_errors.append(error)
            
            self.errors = recent_errors

# Глобальный объект логгера ошибок
error_logger = ErrorLogger()
