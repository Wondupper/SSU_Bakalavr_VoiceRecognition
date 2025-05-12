// Определяем глобальную функцию логирования ошибок
window.logErrorToSystem = function(error, module = "frontend", location = window.location.pathname) {
    // Логируем ошибку в консоль для отладки
    console.error(`[${module}] [${location}] Ошибка: ${error.toString()}`);
};

// Глобальный обработчик ошибок
window.onerror = function(message, source, lineno, colno, error) {
    logErrorToSystem(message, "global", source + ":" + lineno);
    return false;
};

// Проверка совместимости браузера
function checkBrowserCompatibility() {
    const features = {
        fetch: typeof fetch === 'function',
        FormData: typeof FormData === 'function',
        FileReader: typeof FileReader === 'function',
        Audio: typeof Audio === 'function'
    };
    
    const missingFeatures = Object.entries(features)
        .filter(([, supported]) => !supported)
        .map(([feature]) => feature);
    
    if (missingFeatures.length > 0) {
        const message = `Ваш браузер не поддерживает необходимые функции: ${missingFeatures.join(', ')}. Пожалуйста, обновите ваш браузер.`;
        
        // Создаем элемент для отображения ошибки
        const errorElement = document.createElement('div');
        errorElement.style.backgroundColor = '#f44336';
        errorElement.style.color = 'white';
        errorElement.style.padding = '20px';
        errorElement.style.margin = '20px';
        errorElement.style.borderRadius = '5px';
        errorElement.style.textAlign = 'center';
        errorElement.textContent = message;
        
        // Вставляем в начало body
        document.body.insertBefore(errorElement, document.body.firstChild);
        
        console.error(message);
        return false;
    }
    
    return true;
}

document.addEventListener('DOMContentLoaded', () => {
    // Проверяем совместимость браузера
    if (!checkBrowserCompatibility()) {
        return; // Останавливаем инициализацию при несовместимости
    }
    
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
    
    // Запрашиваем эмоцию дня с сервера
    function fetchDailyEmotion() {
        fetch('/api/daily_emotion')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ошибка сервера при получении эмоции дня');
                }
                return response.json();
            })
            .then(data => {
                targetEmotion = data.emotion;
                
                // Устанавливаем текст эмоции на странице
                if (targetEmotionElement) {
                    targetEmotionElement.textContent = targetEmotion;
                    targetEmotionElement.classList.add('fade-in');
                } else {
                    console.error('Элемент с id "target-emotion" не найден');
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
            const userName = data.identity || 'unknown';
            const emotionMatch = data.match === true;
            const success = data.success === true;
            const detectedEmotion = data.emotion || 'unknown';
            
            // Проверка условий успешной идентификации
            if (success) {
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
                    if (detectedEmotion && detectedEmotion !== 'unknown') {
                        errorMessage += ` Обнаруженная эмоция: ${detectedEmotion}.`;
                    }
                }
                
                // Если есть сообщение от сервера, добавляем его
                if (data.message) {
                    errorMessage = data.message;
                }
                
                showResult(errorMessage, 'error');
                document.body.classList.add('error');
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showResult('Ошибка сервера: ' + error.message, 'error');
            document.body.classList.add('error');
            console.error(error.message);
        });
    }
    
    // Функция отображения результата
    function showResult(message, type) {
        resultMessage.innerHTML = message;
        resultMessage.className = 'result-message ' + type;
        resultContainer.classList.add('fade-in');
    }
});
