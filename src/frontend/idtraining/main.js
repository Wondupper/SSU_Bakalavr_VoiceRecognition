document.addEventListener('DOMContentLoaded', () => {
    // Получаем элементы формы
    const nameInput = document.getElementById('name-input');
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
    
    // Обработчик изменения имени
    nameInput.addEventListener('input', validateForm);
    
    // Обработчик отправки формы
    document.getElementById('voice-training-form').addEventListener('submit', (e) => {
        e.preventDefault();
        submitForm();
    });
    
    // Функция валидации формы
    function validateForm() {
        const isNameValid = nameInput.value.trim().length >= 2;
        const isFileValid = audioFile !== null;
        
        submitButton.disabled = !(isNameValid && isFileValid);
    }
    
    // Функция отправки формы
    function submitForm() {
        if (!audioFile || !nameInput.value.trim()) {
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
        formData.append('name', nameInput.value.trim());
        
        // Отправляем запрос на сервер
        fetch('/api/id_training', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            
            if (data.error) {
                showStatus('Ошибка: ' + data.error, 'error');
                return;
            }
            
            showStatus('Обучение модели успешно завершено!', 'success');
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showStatus('Ошибка сервера: ' + error.message, 'error');
            logErrorToSystem(error.message, "voice_id_training", window.location.pathname);
        });
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
