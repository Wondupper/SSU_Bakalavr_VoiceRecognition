import sys
import traceback
import os

class ErrorLogger:
    def __init__(self):
        # Конструктор максимально упрощен
        pass
    
    def log_error(self, error_message, module=None, context=None):
        """
        Записывает ошибку в консоль
        
        Args:
            error_message: Текст ошибки
            module: Модуль, в котором произошла ошибка
            context: Контекст ошибки
        """
        # Формируем сообщение
        log_message = error_message
        if module or context:
            log_message = f"{module or 'unknown'} - {context or 'unknown'} - {error_message}"
        
        # Вывод в консоль
        print(log_message, file=sys.stderr)
    
    def log_exception(self, e, module=None, context=None, message=None):
        """
        Логирует исключение с информацией о файле и номере строки
        
        Args:
            e: Объект исключения
            module: Модуль, в котором произошла ошибка
            context: Контекст ошибки
            message: Дополнительное сообщение
            
        Returns:
            Словарь с информацией об исключении
        """
        # Получаем информацию об исключении
        exc_type, exc_obj, exc_tb = sys.exc_info()
        
        # Получаем трассировку
        tb = traceback.extract_tb(exc_tb)
        
        # Ищем первый фрейм, где произошло исключение (исключая фреймы из библиотек)
        for frame in tb:
            # Проверяем, что путь к файлу не содержит 'site-packages' или 'lib'
            if 'site-packages' not in frame.filename and '/lib/' not in frame.filename:
                # Этот фрейм из нашего кода
                fname = os.path.basename(frame.filename)
                line_no = frame.lineno
                break
        else:
            # Если не нашли подходящий фрейм, берем последний
            tb_last = tb[-1]
            fname = os.path.basename(tb_last.filename)
            line_no = tb_last.lineno
        
        # Формируем сообщение для консоли
        error_str = f"{fname} - {line_no} - {str(e)}"
        
        # Если предоставлено дополнительное сообщение, добавляем его
        if message:
            log_msg = f"{message}: {error_str}"
        else:
            log_msg = error_str
        
        # Выводим в консоль
        self.log_error(log_msg, module, context)
    
    def error(self, message, module=None, context=None):
        """Логирование ошибки"""
        self.log_error(message, module, context)

# Создаем единственный экземпляр логгера
error_logger = ErrorLogger()
