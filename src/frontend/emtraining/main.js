document.addEventListener('DOMContentLoaded', () => {
    // Получаем элементы формы
    const emotionSelect = document.getElementById('emotion-select');
    const audioFileInput = document.getElementById('audio-file');
    const fileNameDisplay = document.getElementById('file-name');
    const audioPreviewContainer = document.getElementById('audio-preview-container');
    const audioPreview = document.getElementById('audio-preview');
    const resetButton = document.getElementById('reset-button');
    const submitButton = document.getElementById('submit-button');
    const statusMessage = document.getElementById('status-message');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    let audioFile = null;
    
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
    
    // Обработчик сброса выбора файла
    resetButton.addEventListener('click', () => {
        audioFileInput.value = '';
        audioFile = null;
        fileNameDisplay.textContent = '';
        audioPreviewContainer.style.display = 'none';
        audioPreview.src = '';
        validateForm();
    });
    
    // Обработчик изменения выбора эмоции
    emotionSelect.addEventListener('change', validateForm);
    
    // Обработчик отправки формы
    document.getElementById('emotion-training-form').addEventListener('submit', (e) => {
        e.preventDefault();
        submitForm();
    });
    
    // Функция валидации формы
    function validateForm() {
        const isEmotionValid = emotionSelect.value !== '';
        const isFileValid = audioFile !== null;
        
        submitButton.disabled = !(isEmotionValid && isFileValid);
    }
    
    // Функция отправки формы
    function submitForm() {
        if (!audioFile || !emotionSelect.value) {
            return;
        }
        
        // Добавляем проверку размера файла
        if (audioFile.size > 20 * 1024 * 1024) { // 20MB максимум
            showStatus('Ошибка: Размер файла превышает 20МБ', 'error');
            return;
        }
        
        // Показываем индикатор загрузки
        loadingIndicator.style.display = 'block';
        
        // Создаем объект FormData для отправки данных
        const formData = new FormData();
        formData.append('audio', audioFile);
        formData.append('emotion', emotionSelect.value);
        
        // Отправляем запрос на сервер
        fetch('/api/em_training', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                loadingIndicator.style.display = 'none';
                showStatus('Ошибка: ' + data.error, 'error');
                return;
            }
            
            // Проверяем статус обучения (поддерживаем как новый, так и старый формат API)
            if (data.status === 'started' || data.message) {
                showStatus('Обучение модели началось. Это может занять некоторое время...', 'info');
                
                // Начинаем мониторинг прогресса обучения
                startTrainingProgressMonitor();
            } else {
                loadingIndicator.style.display = 'none';
                showStatus('Произошла ошибка при обучении модели', 'error');
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showStatus('Ошибка сервера: ' + error.message, 'error');
            logErrorToSystem(error.message, "emotion_training", window.location.pathname);
        });
    }
    
    // Функция для мониторинга прогресса обучения
    function startTrainingProgressMonitor() {
        const statusContainer = document.createElement('div');
        statusContainer.className = 'training-stats';
        statusContainer.innerHTML = 'Обучение модели началось...';
        
        // Добавляем элементы в индикатор загрузки
        loadingIndicator.innerHTML = '<div class="spinner"></div>';
        loadingIndicator.appendChild(statusContainer);
        
        let isTrainingCompleted = false;
        
        // Запускаем интервал для проверки прогресса
        const progressInterval = setInterval(() => {
            // Если обучение уже завершено, прекращаем опрос
            if (isTrainingCompleted) {
                clearInterval(progressInterval);
                return;
            }
            
            fetch('/api/training_progress?model_type=emotion')
                .then(response => {
                    // Проверяем успешность ответа
                    if (!response.ok) {
                        throw new Error(`HTTP error: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    // Проверяем статус обучения
                    if (data.status === 'completed') {
                        // Обучение завершено
                        isTrainingCompleted = true;
                        clearInterval(progressInterval);
                        loadingIndicator.style.display = 'none';
                        showStatus('Обучение модели успешно завершено!', 'success');
                        
                        // Очищаем индикатор для будущих запусков
                        setTimeout(() => {
                            loadingIndicator.innerHTML = '<div class="spinner"></div>';
                        }, 1000);
                    } else if (data.status === 'error') {
                        // Ошибка обучения
                        isTrainingCompleted = true;
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
                    // Если обучение завершено, сервер может перестать отвечать на запросы
                    // прогресса, что вызовет ошибку - это не всегда плохо
                    if (isTrainingCompleted) {
                        return;
                    }
                    
                    // Проверяем, нет ли признаков успешного завершения обучения
                    // в сообщении об ошибке
                    const errorMessage = error.toString().toLowerCase();
                    if (errorMessage.includes('not found') || 
                        errorMessage.includes('no training in progress')) {
                        // Вероятно, обучение завершилось успешно
                        isTrainingCompleted = true;
                        clearInterval(progressInterval);
                        loadingIndicator.style.display = 'none';
                        showStatus('Обучение модели успешно завершено!', 'success');
                        
                        // Очищаем индикатор для будущих запусков
                        setTimeout(() => {
                            loadingIndicator.innerHTML = '<div class="spinner"></div>';
                        }, 1000);
                    } else {
                        console.error('Ошибка при проверке прогресса:', error);
                    }
                });
        }, 1000);
    }
    
    // Функция отображения статуса
    function showStatus(message, type) {
        // Очищаем все существующие классы статуса
        statusMessage.classList.remove('success', 'error', 'info');
        
        // Устанавливаем текст и добавляем нужный класс
        statusMessage.textContent = message;
        statusMessage.classList.add(type);
        
        // Убеждаемся, что сообщение видимо
        statusMessage.style.display = 'block';
    }
});
