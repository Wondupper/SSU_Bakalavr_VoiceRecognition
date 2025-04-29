import sys
import os
import inspect

class InfoLogger:
    def __init__(self):
        # Конструктор максимально упрощен
        pass
    
    def log_info(self, info_message, module=None):
        """
        Записывает информационное сообщение в консоль
        
        Args:
            info_message: Информационный текст
            module: Модуль, из которого вызывается логирование (опционально)
        """
        # Получаем имя файла, откуда был вызван метод
        frame = inspect.stack()[1]
        filename = os.path.basename(frame.filename)
        
        # Если модуль указан явно, используем его
        if module:
            filename = module
            
        # Формируем сообщение
        log_message = f"[INFO] - {filename} - {info_message}"
        
        # Вывод в консоль
        print(log_message)
    
    def info(self, message, module=None):
        """Логирование информационного сообщения"""
        self.log_info(message, module)

# Создаем единственный экземпляр логгера
info_logger = InfoLogger() 