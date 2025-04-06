import threading
from collections import deque
import time

class ErrorLogger:
    def __init__(self, max_errors=100):
        self.errors = deque(maxlen=max_errors)
        self._lock = threading.Lock()
    
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

# Глобальный объект логгера ошибок
error_logger = ErrorLogger()
