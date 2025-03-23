/**
 * Конфигурационные параметры для frontend части приложения
 */

// Параметры для страницы обучения (FP2)
export const TRAINING_CONFIG = {
    MAX_AUDIOFILES_COUNT: 10,      // Максимальное количество файлов
    MAX_AUDIOFILES_SIZE: 104857600, // 100 МБ в байтах
    MAX_NAME_LENGTH: 10,           // Максимальная длина имени
    ALLOWED_AUDIO_FORMAT: 'audio/wav', // Разрешенный формат аудио
    VALIDATION_MESSAGES: {
        MAX_FILES_EXCEEDED: 'Превышено максимальное количество файлов (максимум 10)',
        MAX_SIZE_EXCEEDED: 'Превышен максимальный общий размер файлов (максимум 100 МБ)',
        INVALID_FORMAT: 'Поддерживаются только WAV файлы',
        INVALID_NAME: 'Имя должно содержать только латинские буквы и быть не длиннее 10 символов'
    }
};

// Параметры для страницы идентификации (FP3)
export const IDENTIFICATION_CONFIG = {
    INPUT_AUDIO_LENGTH: 5,     // Длина аудио в секундах
    INPUT_AUDIO_SIZE: 1048576, // 1 МБ в байтах
    ALLOWED_AUDIO_FORMAT: 'audio/wav',
    VALIDATION_MESSAGES: {
        INVALID_LENGTH: 'Длительность аудио должна быть 5 секунд',
        MAX_SIZE_EXCEEDED: 'Размер файла не должен превышать 1 МБ',
        INVALID_FORMAT: 'Поддерживаются только WAV файлы'
    }
};

// Параметры для записи аудио
export const AUDIO_RECORDING_CONFIG = {
    TRAINING_DURATION: 3000,    // 3 секунды для обучения
    IDENTIFY_DURATION: 5000,    // 5 секунд для идентификации
    MIME_TYPE: 'audio/wav',
    SAMPLE_RATE: 16000
};

// Список доступных эмоций
export const EMOTIONS = ['anger', 'joy', 'sadness'];

// Состояния обработки
export const PROCESSING_STATES = {
    IDLE: 'idle',
    RECORDING: 'recording',
    PROCESSING: 'processing',
    SUCCESS: 'success',
    ERROR: 'error'
};

// Цвета для отображения результатов
export const RESULT_COLORS = {
    SUCCESS: '#2ecc71',  // Зеленый
    ERROR: '#e74c3c'     // Красный
}; 