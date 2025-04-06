document.addEventListener('DOMContentLoaded', () => {
    const targetEmotionElement = document.getElementById('target-emotion');
    const audioFileInput = document.getElementById('audio-file');
    const fileNameDisplay = document.getElementById('file-name');
    const audioPreviewContainer = document.getElementById('audio-preview-container');
    const audioPreview = document.getElementById('audio-preview');
    const resetButton = document.getElementById('reset-button');
    const submitButton = document.getElementById('submit-button');
    const resultContainer = document.getElementById('result-container');
    const resultMessage = document.getElementById('result-message');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    let audioFile = null;
    let targetEmotion = null;
    
    // Запрашиваем эмоцию дня с сервера вместо случайной генерации
    function fetchDailyEmotion() {
        fetch('/api/daily_emotion')
            .then(response => response.json())
            .then(data => {
                targetEmotion = data.emotion;
                
                // Устанавливаем текст эмоции на странице
                if (targetEmotionElement) {
                    targetEmotionElement.textContent = targetEmotion;
                    targetEmotionElement.classList.add('fade-in');
                } else {
                    logErrorToSystem('Элемент с id "target-emotion" не найден', "UI", "identification");
                }
            })
            .catch(error => {
                logErrorToSystem('Ошибка получения эмоции дня: ' + error.message, "API", "identification");
                // Запасной вариант - используем "радость" по умолчанию
                targetEmotion = "радость";
                if (targetEmotionElement) {
                    targetEmotionElement.textContent = targetEmotion;
                }
            });
    }
    
    // Сразу запрашиваем эмоцию при загрузке страницы
    fetchDailyEmotion();
    
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
            
            submitButton.disabled = false;
        }
    });
    
    // Обработчик сброса файла
    resetButton.addEventListener('click', () => {
        audioFileInput.value = '';
        audioFile = null;
        fileNameDisplay.textContent = '';
        audioPreviewContainer.style.display = 'none';
        audioPreview.src = '';
        submitButton.disabled = true;
    });
    
    // Обработчик отправки формы
    submitButton.addEventListener('click', submitForm);
    
    // Функция отправки формы
    function submitForm() {
        if (!audioFile) {
            return;
        }
        
        // Добавляем проверку размера файла
        if (audioFile.size > 20 * 1024 * 1024) { // 20MB максимум
            showResult('Ошибка: Размер файла превышает 20МБ', 'error');
            return;
        }
        
        // Сбрасываем предыдущие результаты
        document.body.classList.remove('success', 'error');
        resultContainer.style.display = 'none';
        
        // Показываем индикатор загрузки
        loadingIndicator.style.display = 'block';
        
        // Создаем объект FormData для отправки данных
        const formData = new FormData();
        formData.append('audio', audioFile);
        formData.append('expected_emotion', targetEmotion);
        
        // Отправляем запрос на сервер
        fetch('/api/identify', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            resultContainer.style.display = 'block';
            
            if (data.error) {
                showResult('Ошибка: ' + data.error, 'error');
                return;
            }
            
            // Обработка результата
            const userName = data.user;
            const emotionMatch = data.emotion_match;
            
            // Проверка условий успешной идентификации
            if (userName !== 'unknown' && emotionMatch) {
                // Успешная идентификация
                showResult(`Идентификация успешна! Ваше имя: <span class="result-username">${userName}</span>`, 'success');
                document.body.classList.add('success');
            } else {
                // Неуспешная идентификация
                let errorMessage = 'Идентификация не удалась. ';
                
                if (userName === 'unknown') {
                    errorMessage += 'Система не смогла распознать ваш голос. ';
                }
                
                if (!emotionMatch) {
                    errorMessage += `Эмоция в аудиозаписи не соответствует заданной (${targetEmotion}).`;
                    
                    // Более четкое отображение информации об эмоциях
                    if (data.error_emotion) {
                        errorMessage += ` ${data.error_emotion}.`;
                    } else if (data.detected_emotion) {
                        errorMessage += ` Обнаруженная эмоция: ${data.detected_emotion}.`;
                    }
                }
                
                showResult(errorMessage, 'error');
                document.body.classList.add('error');
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showResult('Ошибка сервера: ' + error.message, 'error');
            document.body.classList.add('error');
            logErrorToSystem(error.message, "api_request", "identification");
        });
    }
    
    // Функция отображения результата
    function showResult(message, type) {
        resultMessage.innerHTML = message;
        resultMessage.className = 'result-message ' + type;
        resultContainer.classList.add('fade-in');
    }
});
