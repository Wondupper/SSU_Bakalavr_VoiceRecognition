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
    
    // Функция проверки валидности формы
    function validateForm() {
        const username = usernameInput.value.trim();
        submitButton.disabled = !(username && audioFile);
    }
    
    // Функция отправки формы
    function submitForm() {
        const username = usernameInput.value.trim();
        
        if (!username || !audioFile) {
            showStatus('Заполните все поля', 'error');
            return;
        }
        
        // Показываем индикатор загрузки
        loadingIndicator.style.display = 'block';
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
            loadingIndicator.style.display = 'none';
            
            if (data.error) {
                showStatus(data.error, 'error');
            } else {
                showStatus(data.message, 'success');
                // Сбрасываем форму после успешной отправки
                usernameInput.value = '';
                resetButton.click();
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            showStatus('Ошибка сервера: ' + error.message, 'error');
        });
    }
    
    // Функция отображения статуса
    function showStatus(message, type) {
        statusMessage.textContent = message;
        statusMessage.className = 'status-message ' + type;
    }
});
