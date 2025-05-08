import os
import inspect

class InfoLogger:
    def __init__(self):
        # Конструктор максимально упрощен
        pass
    
    def log_info(self, info_message, module=None):
        """
        Запись информационного сообщения в консоль
        
        Args:
            info_message: Информационный текст
            module: Модуль, из которого вызывается логирование (опционально)
        """
        # Получаем стек вызовов
        stack = inspect.stack()
        
        # Определяем имя файла, из которого фактически вызван логгер
        calling_filename = None
        for frame_info in stack:
            frame_filename = os.path.basename(frame_info.filename)
            # Если имя текущего файла не info_logger.py, это и есть вызывающий файл
            if frame_filename != os.path.basename(__file__):
                calling_filename = frame_filename
                break
        
        # Если не нашли вызывающий файл, используем текущий
        if not calling_filename:
            calling_filename = os.path.basename(__file__)
        
        # Если модуль указан явно, используем его
        if module:
            calling_filename = module
            
        # Формируем сообщение
        log_message = f"[INFO] - {calling_filename} - {info_message}"
        
        # Вывод в консоль
        print(log_message)
    
    def info(self, message, module=None):
        """Логирование информационного сообщения"""
        self.log_info(message, module)

# Создаем единственный экземпляр логгера
info_logger = InfoLogger() 