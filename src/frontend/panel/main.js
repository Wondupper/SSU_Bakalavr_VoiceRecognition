document.addEventListener('DOMContentLoaded', () => {
    // Кнопки для модели идентификации
    const resetVoiceModelBtn = document.getElementById('reset-voice-model');
    const saveVoiceModelBtn = document.getElementById('save-voice-model');
    const loadVoiceModelInput = document.getElementById('load-voice-model');
    
    // Кнопки для модели эмоций
    const resetEmotionModelBtn = document.getElementById('reset-emotion-model');
    const saveEmotionModelBtn = document.getElementById('save-emotion-model');
    const loadEmotionModelInput = document.getElementById('load-emotion-model');
    
    // Элементы статуса
    const voiceModelStatus = document.getElementById('voice-model-status');
    const emotionModelStatus = document.getElementById('emotion-model-status');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Обработчики для модели идентификации по голосу
    resetVoiceModelBtn.addEventListener('click', () => resetModel('voice_id'));
    saveVoiceModelBtn.addEventListener('click', () => saveModel('voice_id'));
    loadVoiceModelInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            loadModel('voice_id', e.target.files[0]);
        }
    });
    
    // Обработчики для модели распознавания эмоций
    resetEmotionModelBtn.addEventListener('click', () => resetModel('emotion'));
    saveEmotionModelBtn.addEventListener('click', () => saveModel('emotion'));
    loadEmotionModelInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            loadModel('emotion', e.target.files[0]);
        }
    });
    
    // Функция сброса модели
    function resetModel(modelType) {
        if (!confirm(`Вы уверены, что хотите сбросить модель ${modelType === 'voice_id' ? 'идентификации по голосу' : 'распознавания эмоций'}?`)) {
            return;
        }
        
        showLoading(true);
        
        fetch('/api/panel/reset_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType })
        })
        .then(response => response.json())
        .then(data => {
            showLoading(false);
            
            if (data.error) {
                showStatus(modelType, data.error, 'error');
            } else {
                showStatus(modelType, data.message, 'success');
            }
        })
        .catch(error => {
            showLoading(false);
            showStatus(modelType, 'Ошибка сервера: ' + error.message, 'error');
        });
    }
    
    // Функция сохранения модели
    function saveModel(modelType) {
        showLoading(true);
        
        fetch('/api/panel/save_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType })
        })
        .then(response => response.json())
        .then(data => {
            showLoading(false);
            
            if (data.error) {
                showStatus(modelType, data.error, 'error');
            } else {
                showStatus(modelType, data.message, 'success');
            }
        })
        .catch(error => {
            showLoading(false);
            showStatus(modelType, 'Ошибка сервера: ' + error.message, 'error');
        });
    }
    
    // Функция загрузки модели
    function loadModel(modelType, file) {
        // Создаем объект FormData для отправки файла
        const formData = new FormData();
        formData.append('model_file', file);
        
        showLoading(true);
        
        // Сначала загружаем файл на сервер
        fetch('/api/panel/upload_model', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showLoading(false);
                showStatus(modelType, data.error, 'error');
            } else {
                // Если загрузка успешна, вызываем API загрузки модели с путем к файлу
                const filePath = data.file_path;
                
                fetch('/api/panel/load_model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model_type: modelType,
                        file_path: filePath
                    })
                })
                .then(response => response.json())
                .then(data => {
                    showLoading(false);
                    if (data.error) {
                        showStatus(modelType, data.error, 'error');
                    } else {
                        showStatus(modelType, data.message, 'success');
                    }
                })
                .catch(error => {
                    showLoading(false);
                    showStatus(modelType, 'Ошибка сервера: ' + error.message, 'error');
                });
            }
        })
        .catch(error => {
            showLoading(false);
            showStatus(modelType, 'Ошибка загрузки файла: ' + error.message, 'error');
        });
    }
    
    // Функция отображения статуса
    function showStatus(modelType, message, type) {
        const statusElement = modelType === 'voice_id' ? voiceModelStatus : emotionModelStatus;
        statusElement.textContent = message;
        statusElement.className = 'status-message ' + type;
    }
    
    // Функция отображения/скрытия индикатора загрузки
    function showLoading(show) {
        loadingIndicator.style.display = show ? 'block' : 'none';
    }
});
