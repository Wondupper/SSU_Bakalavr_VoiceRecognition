const config = {
    // Конфигурация для страницы обучения (FP2)
    training: {
        maxAudioFilesCount: 10,
        maxAudioFilesSize: 104857600, // 100 МБ в байтах
        maxNameLength: 10,
        allowedAudioFormat: 'audio/wav',
        validationMessages: {
            maxFilesExceeded: 'Превышено максимальное количество файлов',
            maxSizeExceeded: 'Превышен максимальный размер файлов',
            invalidFormat: 'Неверный формат файла',
            invalidName: 'Имя может содержать только латинские буквы'
        }
    },

    // Конфигурация для страницы идентификации (FP3)
    identification: {
        inputAudioLength: 5, // секунды
        inputAudioSize: 1048576, // 1 МБ в байтах
        allowedAudioFormat: 'audio/wav',
        validationMessages: {
            maxLengthExceeded: 'Превышена максимальная длина записи',
            maxSizeExceeded: 'Превышен максимальный размер файла',
            invalidFormat: 'Неверный формат файла'
        }
    },

    // Конфигурация записи аудио
    audioRecording: {
        trainingDuration: 3000, // 3 секунды в миллисекундах
        identifyDuration: 5000, // 5 секунд в миллисекундах
        mimeType: 'audio/wav',
        sampleRate: 16000
    },

    // Поддерживаемые эмоции
    emotions: ['anger', 'joy', 'sadness'],

    // Состояния обработки
    processingStates: {
        idle: 'idle',
        recording: 'recording',
        processing: 'processing',
        success: 'success',
        error: 'error'
    },

    // Цвета для результатов
    resultColors: {
        success: '#2ecc71',
        error: '#e74c3c'
    }
};

export default config; 