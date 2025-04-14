import os
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
from functools import lru_cache  # Импорт декоратора lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # Поддержка многопоточности
import multiprocessing
from backend.api.error_logger import error_logger
import sys
import traceback

# Константы
AUDIO_FRAGMENT_LENGTH = 3  # длина фрагмента в секундах
SAMPLE_RATE = 16000  # частота дискретизации
# Константы для шумоподавления
NOISE_PERCENTILE = 10  # Персентиль для определения порога шума
MIN_GAIN = 0.1  # Минимальный коэффициент усиления

# Определяем оптимальное количество процессов/потоков
# Ограничиваем максимальное количество процессоров для снижения нагрузки на память
N_JOBS = max(1, min(2, multiprocessing.cpu_count() - 1))  # Не более 2 процессов
MAX_MEMORY_PER_PROCESS = 512 * 1024 * 1024  # Максимум 512MB на процесс

def process_audio(audio_file):
    """
    Основная функция обработки аудиофайла
    Максимально упрощенная и надежная
    """
    try:
        # Проверка объекта файла
        if not audio_file:
            raise ValueError("Аудиофайл не предоставлен")
        
        # Сохраняем позицию файла
        original_position = audio_file.tell()
        
        # Чтение аудиофайла
        audio_bytes = audio_file.read()
        
        # Возвращаем файл в исходное положение
        audio_file.seek(original_position)
        
        # Проверка размера файла
        if len(audio_bytes) == 0:
            raise ValueError("Пустой аудиофайл")
        
        # Проверка формата файла
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        if file_extension not in ['.wav', '.mp3', '.ogg', '.flac']:
            raise ValueError(f"Неподдерживаемый формат аудиофайла: {file_extension}")
        
        # Логируем начало обработки
        error_logger.log_error(
            f"Обработка аудиофайла: {audio_file.filename}",
            "audio", "process_audio"
        )
        
        try:
            # Максимально надежная загрузка аудио
            audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, res_type='kaiser_fast')
        except Exception as e:
            raise ValueError(f"Не удалось декодировать аудиофайл: {str(e)}")
        
        # Проверка на пустое аудио
        if len(audio_data) == 0 or np.all(audio_data == 0):
            raise ValueError("Аудиофайл не содержит данных")
        
        # Логируем информацию о загруженном файле
        error_logger.log_error(
            f"Аудио загружено: длина {len(audio_data)}, частота {sr}",
            "audio", "process_audio"
        )
        
        # Версия обработки без сложных алгоритмов для максимальной надежности
        try:
            # Шаг 1: Нормализация (самая надежная операция)
            audio_data = normalize_audio(audio_data)
            
            # Проверка результата нормализации
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                error_logger.log_error(
                    "Обнаружены недопустимые значения после нормализации",
                    "audio", "process_audio"
                )
                # Загружаем аудио заново и пропускаем обработку
                audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
                audio_data = np.nan_to_num(audio_data)  # Заменяем NaN и Inf на числа
            else:
                # Шаг 2: Простое удаление шума (используем надежную библиотечную функцию)
                audio_data = enhanced_noise_removal(audio_data)
                
                # Проверка результата удаления шума
                if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                    error_logger.log_error(
                        "Обнаружены недопустимые значения после удаления шума",
                        "audio", "process_audio"
                    )
                    # Загружаем аудио заново и пропускаем дальнейшую обработку
                    audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
                    audio_data = normalize_audio(audio_data)
                else:
                    # Шаг 3: Удаление тишины
                    audio_data = enhanced_silence_removal(audio_data, sr)
                    
                    # Проверка результата удаления тишины
                    if np.isnan(audio_data).any() or np.isinf(audio_data).any() or len(audio_data) < sr * 0.1:
                        error_logger.log_error(
                            f"Проблема после удаления тишины: длина={len(audio_data)}",
                            "audio", "process_audio"
                        )
                        # Возвращаемся к нормализованному аудио
                        audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
                        audio_data = normalize_audio(audio_data)
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            line_no = exc_tb.tb_lineno
            print(f"{fname} - {line_no} - {str(e)}")
            error_logger.log_error(
                f"Ошибка при обработке аудио: {str(e)}",
                "audio", "process_audio"
            )
            # Используем прямую загрузку без обработки при любой ошибке
            audio_data, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE)
        
        # Проверка размера аудио перед разделением
        min_length = int(AUDIO_FRAGMENT_LENGTH * sr * 0.25)
        if len(audio_data) < min_length:
            error_logger.log_error(
                f"Аудио слишком короткое для разделения: {len(audio_data)} < {min_length}",
                "audio", "process_audio"
            )
            raise ValueError(f"Аудио слишком короткое для обработки: {len(audio_data) / sr:.2f} секунд")
        
        # Разделение на фрагменты максимально надежным способом
        audio_fragments = improved_split_audio(audio_data, sr)
        
        if not audio_fragments:
            error_logger.log_error(
                "Не удалось получить фрагменты, создаем один фрагмент вручную",
                "audio", "process_audio"
            )
            # Создаем хотя бы один фрагмент, если разделение не удалось
            fragment_size = int(AUDIO_FRAGMENT_LENGTH * sr)
            fragment = np.zeros(fragment_size, dtype=np.float32)
            
            # Копируем данные с начала аудио, чтобы избежать проблем с размерностями
            copy_size = min(len(audio_data), fragment_size)
            fragment[:copy_size] = audio_data[:copy_size]
            
            audio_fragments = [fragment]
        
        # Логируем успешное завершение
        error_logger.log_error(
            f"Обработка завершена: получено {len(audio_fragments)} фрагментов",
            "audio", "process_audio"
        )
        
        return audio_fragments
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        error_message = f"Критическая ошибка обработки аудио: {str(e)}"
        error_logger.log_error(error_message, "audio", "process_audio")
        raise ValueError(error_message)

def normalize_audio(audio_data):
    """
    Нормализация амплитуды аудио для улучшения качества обработки
    Функция уже оптимальная, так как использует векторизованные операции numpy
    """
    # Проверка на пустые данные
    if len(audio_data) == 0:
        return audio_data
    
    # Нормализация до диапазона [-1, 1]
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        return audio_data / max_abs
    return audio_data

def enhanced_noise_removal(audio_data):
    """
    Удаление шума с использованием стандартных методов спектральной обработки
    """
    # Проверка на пустые данные
    if len(audio_data) == 0:
        return audio_data
    
    # Для очень коротких аудио пропускаем шумоподавление
    if len(audio_data) < 2048:
        return audio_data
    
    try:
        # Простой и надежный подход к шумоподавлению на основе спектрального вычитания
        
        # Шаг 1: Взять STFT с помощью librosa (надежный способ)
        # Уменьшаем размер n_fft для предотвращения исчерпания памяти
        n_fft = 1024  # Уменьшено с 2048
        hop_length = 256  # Уменьшено с 512
        
        # Ограничиваем длину аудио для обработки (макс. 10 секунд)
        max_samples = 10 * SAMPLE_RATE
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        
        # Шаг 2: Получить спектрограмму мощности
        S = np.abs(D) ** 2
        
        # Шаг 3: Оценить уровень шума как нижний процентиль спектра
        noise_threshold = np.median(np.percentile(S, 20, axis=1))
        
        # Шаг 4: Построить мягкую маску для подавления шума
        mask = np.maximum(0, 1 - (noise_threshold / (S + 1e-10)))
        
        # Шаг 5: Применить маску и восстановить сигнал с фазой исходного
        D_denoised = D * mask[:, np.newaxis]
        
        # Шаг 6: Выполнить обратный STFT для получения очищенного сигнала
        audio_denoised = librosa.istft(D_denoised, hop_length=hop_length, length=len(audio_data))
        
        # Нормализация до исходной громкости
        if np.max(np.abs(audio_denoised)) > 0:
            gain = np.max(np.abs(audio_data)) / np.max(np.abs(audio_denoised))
            audio_denoised = audio_denoised * gain
        
        # Проверка на наличие недопустимых значений
        if np.isnan(audio_denoised).any() or np.isinf(audio_denoised).any():
            error_logger.log_error(
                "Обнаружены NaN/Inf значения после шумоподавления, возвращаем исходный сигнал",
                "audio", "enhanced_noise_removal"
            )
            return audio_data
        
        return audio_denoised
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(
            f"Ошибка при удалении шума: {str(e)}", 
            "audio", "enhanced_noise_removal"
        )
        # В случае ошибки возвращаем исходный сигнал
        return audio_data

def enhanced_silence_removal(audio_data, sr):
    """
    Удаление тишины с использованием стандартных функций библиотеки librosa
    """
    # Проверка на пустые данные
    if len(audio_data) == 0:
        return audio_data
    
    # Для очень коротких аудио сразу возвращаем исходный сигнал
    if len(audio_data) < sr * 0.5:  # Меньше 0.5 секунды
        return audio_data
    
    try:
        # Используем стандартные функции librosa для удаления тишины
        # Этот подход значительно надежнее самописного кода
        
        # Определяем порог тишины как небольшой процент от максимальной амплитуды
        threshold = 0.05 * np.max(np.abs(audio_data))
        
        # Используем librosa.effects.split для нахождения непустых интервалов
        non_silent_intervals = librosa.effects.split(
            audio_data, 
            top_db=20,  # Стандартное значение порога в децибелах
            frame_length=512, 
            hop_length=128
        )
        
        # Если не найдены непустые интервалы, возвращаем оригинал
        if len(non_silent_intervals) == 0:
            return audio_data
        
        # Для безопасности обработаем случай, когда функция возвращает пустой массив
        error_logger.log_error(
            f"Найдено {len(non_silent_intervals)} непустых интервалов",
            "audio", "enhanced_silence_removal"
        )
        
        # Собираем непустые части аудио в один массив
        non_silent_segments = []
        for start, end in non_silent_intervals:
            # Добавляем контекст - небольшой участок перед и после сегмента
            padded_start = max(0, start - int(0.05 * sr))  # 50 мс контекста
            padded_end = min(len(audio_data), end + int(0.05 * sr))
            segment = audio_data[padded_start:padded_end].copy()
            if len(segment) > 0:
                non_silent_segments.append(segment)
        
        # Если нет сегментов, возвращаем исходные данные
        if not non_silent_segments:
            return audio_data
        
        # Объединяем непустые сегменты
        try:
            result = np.concatenate(non_silent_segments)
            
            # Проверяем длину результата
            if len(result) < len(audio_data) * 0.1:
                # Если удалено более 90% аудио, это может быть ошибка
                error_logger.log_error(
                    f"Удалено более 90% аудио, возвращаем оригинал. Было: {len(audio_data)}, стало: {len(result)}",
                    "audio", "enhanced_silence_removal"
                )
                return audio_data
            
            return result
        except Exception as e:
            error_logger.log_error(
                f"Ошибка при объединении сегментов: {str(e)}",
                "audio", "enhanced_silence_removal"
            )
            return audio_data
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(
            f"Ошибка при удалении тишины: {str(e)}",
            "audio", "enhanced_silence_removal"
        )
        # В случае любой ошибки возвращаем исходный сигнал
        return audio_data

def improved_split_audio(audio_data, sr):
    """
    Максимально простая и надежная функция разделения аудио на фрагменты
    """
    # Проверка на пустые данные
    if audio_data is None or len(audio_data) == 0:
        error_logger.log_error("Пустой аудиосигнал при разделении на фрагменты", "audio", "improved_split_audio")
        return []
    
    try:
        # Размер одного фрагмента в отсчетах
        fragment_size = int(AUDIO_FRAGMENT_LENGTH * sr)
        
        # Создаем максимально простой и надежный код для разделения аудио
        fragments = []
        
        # Если аудио короче минимального размера (25% от фрагмента), возвращаем пустой список
        if len(audio_data) < fragment_size * 0.25:
            error_logger.log_error(
                f"Аудио слишком короткое: {len(audio_data)} отсчетов, минимум: {int(fragment_size * 0.25)}",
                "audio", "improved_split_audio"
            )
            return []
        
        # Если аудио короче полного фрагмента, но достаточно длинное - дополняем нулями
        if len(audio_data) < fragment_size:
            error_logger.log_error(
                f"Аудио короче фрагмента, дополняем нулями: {len(audio_data)} -> {fragment_size}",
                "audio", "improved_split_audio"
            )
            fragment = np.zeros(fragment_size, dtype=np.float32)
            fragment[:len(audio_data)] = audio_data
            fragments.append(fragment)
            return fragments
        
        # Для длинного аудио - простое разделение без перекрытия
        total_fragments = len(audio_data) // fragment_size
        for i in range(total_fragments):
            start = i * fragment_size
            end = start + fragment_size
            # Создаем копию данных для предотвращения проблем с указателями
            fragments.append(audio_data[start:end].copy())
        
        # Обрабатываем остаток, если он достаточно большой
        remainder = len(audio_data) % fragment_size
        if remainder >= fragment_size * 0.25:
            # Создаем дополненный нулями фрагмент
            last_fragment = np.zeros(fragment_size, dtype=np.float32)
            last_fragment[:remainder] = audio_data[-remainder:].copy()
            fragments.append(last_fragment)
        
        # Проверяем, создан ли хотя бы один фрагмент
        if not fragments:
            error_logger.log_error(
                "Не удалось создать ни одного фрагмента, пробуем создать хотя бы один",
                "audio", "improved_split_audio"
            )
            # Гарантируем, что вернем хотя бы один фрагмент, если аудио достаточно длинное
            if len(audio_data) >= fragment_size * 0.25:
                fragment = np.zeros(fragment_size, dtype=np.float32)
                fragment[:min(len(audio_data), fragment_size)] = audio_data[:min(len(audio_data), fragment_size)].copy()
                fragments.append(fragment)
        
        # Логируем результат
        error_logger.log_error(
            f"Аудио разделено на {len(fragments)} фрагментов",
            "audio", "improved_split_audio"
        )
        
        return fragments
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        print(f"{fname} - {line_no} - {str(e)}")
        
        error_logger.log_error(
            f"Ошибка при разделении аудио: {str(e)}",
            "audio", "improved_split_audio"
        )
        # В случае исключения пытаемся вернуть хотя бы минимальный результат
        try:
            fragment_size = int(AUDIO_FRAGMENT_LENGTH * sr)
            if len(audio_data) >= fragment_size * 0.25:
                fragment = np.zeros(fragment_size, dtype=np.float32)
                fragment[:min(len(audio_data), fragment_size)] = audio_data[:min(len(audio_data), fragment_size)]
                return [fragment]
        except Exception as inner_e:
            # Добавляем информацию о внутреннем исключении
            inner_exc_type, inner_exc_obj, inner_exc_tb = sys.exc_info()
            inner_fname = os.path.basename(inner_exc_tb.tb_frame.f_code.co_filename)
            inner_line_no = inner_exc_tb.tb_lineno
            print(f"{inner_fname} - {inner_line_no} - {str(inner_e)}")
            pass
        
        return []
