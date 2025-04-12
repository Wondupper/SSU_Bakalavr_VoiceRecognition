document.addEventListener('DOMContentLoaded', () => {
    const usernameInput = document.getElementById('username');
    const audioFileInput = document.getElementById('audio-file');
    const fileNameDisplay = document.getElementById('file-name');
    const audioPreviewContainer = document.getElementById('audio-preview-container');
    const audioPreview = document.getElementById('audio-preview');
    const resetButton = document.getElementById('reset-button');
    const submitButton = document.getElementById('submit-button');
    const statusMessage = document.getElementById('status-message');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Элементы управления моделью
    const resetVoiceModelBtn = document.getElementById('reset-voice-model');
    const saveVoiceModelBtn = document.getElementById('save-voice-model');
    const loadVoiceModelInput = document.getElementById('load-voice-model');
    const voiceModelStatus = document.getElementById('voice-model-status');
    
    let audioFile = null;
    
    // Обработчик изменения имени пользователя
    usernameInput.addEventListener('input', validateForm);
    
    // Обработчик выбора файла
    audioFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            audioFile = file;
            fileNameDisplay.textContent = file.name;
            
            // Предпрослушивание аудио
            const audioURL = URL.createObjectURL(file);
            audioPreview.src = audioURL;
            audioPreviewContainer.style.display = 'block';
            
            validateForm();
        }
    });
    
    // Обработчик сброса файла
    resetButton.addEventListener('click', () => {
        audioFileInput.value = '';
        audioFile = null;
        fileNameDisplay.textContent = '';
        audioPreviewContainer.style.display = 'none';
        audioPreview.src = '';
        validateForm();
    });
    
    // Обработчик отправки формы
    submitButton.addEventListener('click', submitForm);
    
    // Обработчики для управления моделью
    resetVoiceModelBtn.addEventListener('click', () => resetModel('voice_id'));
    saveVoiceModelBtn.addEventListener('click', () => saveModel('voice_id'));
    loadVoiceModelInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            loadModel('voice_id', e.target.files[0]);
        }
    });
    
    // Функция проверки валидности формы
    function validateForm() {
        const username = usernameInput.value.trim();
        const hasFile = audioFileInput.files.length > 0;
        
        submitButton.disabled = !(username && hasFile);
    }
    
    // Функция отправки формы
    function submitForm() {
        const username = usernameInput.value.trim();
        
        if (!username || !audioFile) {
            showStatus('Заполните все поля', 'error');
            return;
        }
        
        // Показываем индикатор загрузки
        loadingIndicator.style.display = 'flex';
        statusMessage.textContent = '';
        statusMessage.className = 'status-message';
        
        // Создаем объект FormData для отправки данных
        const formData = new FormData();
        formData.append('name', username);
        formData.append('audio', audioFile);
        
        // Отправляем запрос на сервер
        fetch('/api/id_training', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                loadingIndicator.style.display = 'none';
                showStatus(data.error, 'error');
            } else {
                showStatus(data.message, 'success');
                
                // Начинаем мониторинг прогресса обучения
                startTrainingProgressMonitor();
                
                // Сбрасываем форму после успешной отправки
                usernameInput.value = '';
                resetButton.click();
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showStatus('Ошибка сервера: ' + error.message, 'error');
            logErrorToSystem(error.message, "api_request", "training");
        });
    }
    
    // Функция для мониторинга прогресса обучения
    function startTrainingProgressMonitor() {
        // Устанавливаем индикатор загрузки
        const loadingText = document.createElement('p');
        loadingText.textContent = 'Идет обучение модели...';
        loadingIndicator.appendChild(loadingText);
        
        // Индикатор прогресса
        const progressContainer = document.createElement('div');
        progressContainer.style.width = '100%';
        progressContainer.style.height = '20px';
        progressContainer.style.backgroundColor = '#f3f3f3';
        progressContainer.style.borderRadius = '10px';
        progressContainer.style.marginTop = '20px';
        progressContainer.style.marginBottom = '10px';
        progressContainer.style.overflow = 'hidden';
        
        const progressBar = document.createElement('div');
        progressBar.style.width = '0%';
        progressBar.style.height = '100%';
        progressBar.style.backgroundColor = '#4CAF50';
        progressBar.style.transition = 'width 0.5s';
        progressContainer.appendChild(progressBar);
        loadingIndicator.appendChild(progressContainer);
        
        // Статус обучения
        const statusContainer = document.createElement('div');
        statusContainer.style.color = 'white';
        statusContainer.style.marginTop = '10px';
        statusContainer.textContent = 'Подготовка данных...';
        loadingIndicator.appendChild(statusContainer);
        
        // Запускаем интервал для проверки прогресса
        const progressInterval = setInterval(() => {
            fetch('/api/training_progress?model_type=voice_id')
                .then(response => response.json())
                .then(data => {
                    // Обновляем прогресс-бар
                    if (data.status === 'training') {
                        const percent = Math.round((data.current_epoch / data.total_epochs) * 100);
                        progressBar.style.width = `${percent}%`;
                        
                        // Вычисляем прошедшее время
                        const elapsedTime = Math.round((Date.now() - data.start_time * 1000) / 1000);
                        const minutes = Math.floor(elapsedTime / 60);
                        const seconds = elapsedTime % 60;
                        
                        // Обновляем статус
                        statusContainer.innerHTML = `
                            Эпоха: ${data.current_epoch} / ${data.total_epochs}<br>
                            Точность: ${(data.accuracy * 100).toFixed(2)}%<br>
                            Ошибка: ${data.loss.toFixed(4)}<br>
                            Прошло времени: ${minutes}м ${seconds}с
                        `;
                    } else if (data.status === 'completed') {
                        // Обучение завершено
                        clearInterval(progressInterval);
                        loadingIndicator.style.display = 'none';
                        showStatus('Обучение модели успешно завершено!', 'success');
                        
                        // Очищаем индикатор для будущих запусков
                        setTimeout(() => {
                            loadingIndicator.innerHTML = '<div class="spinner"></div>';
                        }, 1000);
                    } else if (data.status === 'error') {
                        // Ошибка обучения
                        clearInterval(progressInterval);
                        loadingIndicator.style.display = 'none';
                        showStatus('Ошибка при обучении модели. Пожалуйста, попробуйте еще раз.', 'error');
                        
                        // Очищаем индикатор для будущих запусков
                        setTimeout(() => {
                            loadingIndicator.innerHTML = '<div class="spinner"></div>';
                        }, 1000);
                    }
                })
                .catch(error => {
                    console.error('Ошибка при проверке прогресса:', error);
                });
        }, 1000);
    }
    
    // Функция отображения статуса
    function showStatus(message, type) {
        statusMessage.textContent = message;
        statusMessage.className = 'status-message ' + type;
    }
    
    // Функции управления моделями из панели администрирования
    function resetModel(modelType) {
        if (!confirm(`Вы уверены, что хотите сбросить модель идентификации по голосу?`)) {
            return;
        }
        
        loadingIndicator.style.display = 'block';
        statusMessage.textContent = '';
        statusMessage.className = 'status-message';
        
        fetch('/api/model/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType })
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            
            if (data.error) {
                showModelStatus(data.error, 'error');
            } else {
                showModelStatus(data.message, 'success');
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showModelStatus('Ошибка сервера: ' + error.message, 'error');
            logErrorToSystem(error.message, "model_reset", window.location.pathname);
        });
    }
    
    function saveModel(modelType) {
        loadingIndicator.style.display = 'block';
        statusMessage.textContent = '';
        statusMessage.className = 'status-message';
        
        fetch('/api/model/download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType })
        })
        .then(response => {
            loadingIndicator.style.display = 'none';
            
            if (response.ok) {
                // Если успешно, начинаем скачивание файла
                showModelStatus('Модель успешно сохранена и скачивается...', 'success');
                return response.blob();
            } else {
                // Обрабатываем ошибку
                return response.json().then(data => {
                    throw new Error(data.error || 'Ошибка при скачивании модели');
                });
            }
        })
        .then(blob => {
            // Создаем ссылку для скачивания
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = modelType === 'voice_id' ? 'voice_id_model.zip' : 'emotion_model.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showModelStatus('Ошибка сервера: ' + error.message, 'error');
            logErrorToSystem(error.message, "model_download", window.location.pathname);
        });
    }
    
    function loadModel(modelType, file) {
        // Создаем объект FormData для отправки файла
        const formData = new FormData();
        formData.append('model_file', file);
        formData.append('model_type', modelType);
        
        loadingIndicator.style.display = 'block';
        statusMessage.textContent = '';
        statusMessage.className = 'status-message';
        
        // Загружаем файл на сервер в соответствующую директорию
        fetch('/api/model/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                loadingIndicator.style.display = 'none';
                showModelStatus(data.error, 'error');
                return;
            }
            
            // Если загрузка файла прошла успешно, вызываем API для загрузки модели из файла
            const filePath = data.file_path;
            
            fetch('/api/model/load', {
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
                loadingIndicator.style.display = 'none';
                
                if (data.error) {
                    showModelStatus(data.error, 'error');
                } else {
                    showModelStatus(data.message, 'success');
                }
            })
            .catch(error => {
                loadingIndicator.style.display = 'none';
                showModelStatus('Ошибка сервера: ' + error.message, 'error');
                logErrorToSystem(error.message, "model_load", window.location.pathname);
            });
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showModelStatus('Ошибка загрузки файла: ' + error.message, 'error');
            logErrorToSystem(error.message, "model_upload", window.location.pathname);
        });
    }
    
    function showModelStatus(message, type) {
        voiceModelStatus.textContent = message;
        voiceModelStatus.className = 'status-message ' + type;
    }
});
