"""
Конфигурационный файл для backend части проекта.
Содержит все константы, используемые в различных модулях.
"""

# Общие константы для всего проекта
SAMPLE_RATE = 16000  # Частота дискретизации для всех аудиофайлов
EMOTIONS = ['радость', 'спокойствие']  # Поддерживаемые эмоции
IS_AUGMENTATION_ENABLED = False

# Константы для обработки аудио
AUDIO_FRAGMENT_LENGTH = 5  # длина фрагмента в секундах

# Константы для аугментации
AUGMENTATION = {
    # Константы для работы с аудио и аугментации
    'SPEEDS': [0.5, 0.75, 1.5, 2 ],   # Более тонкая градация коэффициентов скорости
    'DECAYS': [0.3, 0.7],               # Коэффициенты затухания для реверберации. Диапазон значений: 0.1–0.9. Пример: 0.3 – быстрое затухание, лёгкая реверберация, 0.8 – очень длинный «хвост», звук как в большом зале.
    'MASK_PARAMS': [0.05, 0.1],        # Параметры для маскирования по времени (уменьшены). Диапазон значений: 0.05–0.1. Пример: 0.05 – «маска» закрывает 5% временной оси спектра, 0.15 – 15% временной оси.
    'SNR_DBS': [14, 20]              # Уровень шума для добавления шума. Диапазон значений:  0 до 30 дБ. Чем ниже SNR, тем сильнее шум и хуже слышен исходный сигнал.
}

# # Общие константы для моделей
# COMMON_MODELS_PARAMS = {
#     'TRAIN_SPLIT': 0.7,            # Доля тренировочной выборки
#     'EARLY_STOP_PATIENCE': 15,      # Терпение для раннего останова
#     'BATCH_SIZE': 16,            # Размер батча для обучения
#     'VAL_SPLIT': 0.3,            # Доля валидационной выборки
#     'EPOCHS': 10,               # Максимальное количество эпох обучения
#     'PATIENCE': 20,              # Терпение для раннего останова
#     'LEARNING_RATE': 0.0005,      # Начальная скорость обучения
#     'WEIGHT_DECAY': 1e-4,        # L2 регуляризация
#     'SCHEDULER_FACTOR': 0.5,     # Коэффициент для планировщика скорости обучения
#     'SCHEDULER_PATIENCE': 10,     # Терпение для планировщика скорости обучения
#     'MIN_LR': 0.00001
# }

# Общие константы для моделей
COMMON_MODELS_PARAMS = {
    'TRAIN_SPLIT': 0.5,            # Доля тренировочной выборки
    'EARLY_STOP_PATIENCE': 10,      # Терпение для раннего останова
    'BATCH_SIZE': 32,            # Размер батча для обучения
    'VAL_SPLIT': 0.4,            # Доля валидационной выборки
    'EPOCHS': 10,               # Максимальное количество эпох обучения
    'PATIENCE': 10,              # Терпение для раннего останова
    'LEARNING_RATE': 0.01,      # Уменьшенная скорость обучения
    'WEIGHT_DECAY': 1e-3,        # Увеличенная L2 регуляризация
    'SCHEDULER_FACTOR': 0.5,     # Коэффициент для планировщика скорости обучения
    'SCHEDULER_PATIENCE': 10,     # Терпение для планировщика скорости обучения
    'MIN_LR': 0.001,
    'SOFTMAX_TEMPERATURE': 1.5,  # Температура softmax для более "мягких" вероятностей
}

# Константы для модели голосовой идентификации
VOICE_MODEL_PARAMS = {
    'FEATURE_TARGET_LENGTH': 200,  # Целевая длина временного измерения признаков
    'MIN_CONFIDENCE': 0.65,        # Уменьшен порог уверенности для более сбалансированных предсказаний
}

# Константы для модели распознавания эмоций
EMOTIONS_MODEL_PARAMS = {
    'FEATURE_TARGET_LENGTH': 200,  # Целевая длина временного измерения признаков
    'MIN_CONFIDENCE': 0.0,         # Минимальная уверенность для распознавания
}

DATA_EMOTIONS = {             
    'радость': '/home/vano/myprojects/python/SSU_Bakalavr_VoiceRecognition/testing/test_input_audiofiles/emotions/happy.wav',              
    'спокойствие': '/home/vano/myprojects/python/SSU_Bakalavr_VoiceRecognition/testing/test_input_audiofiles/emotions/norm.wav',              
}

DATA_VOICE = {
    'FirstSpeaker': '/home/vano/myprojects/python/SSU_Bakalavr_VoiceRecognition/testing/test_input_audiofiles/voice/first.wav',                       
    'SecondSpeaker': '/home/vano/myprojects/python/SSU_Bakalavr_VoiceRecognition/testing/test_input_audiofiles/voice/second.wav',                       
}
