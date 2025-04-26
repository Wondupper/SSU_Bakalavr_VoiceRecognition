// Глобальный статус для отслеживания обучения моделей
let isModelTraining = false;

// Переменная для отслеживания статуса интервалов
let intervals = {
    statusCheck: null,
    progressCheck: null
};

// Добавим переменную для хранения типа текущего обучения
let currentTrainingType = null;

// Проверка статуса моделей каждые 5 секунд
function checkModelStatus() {
    fetch('/api/status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const wasTraining = isModelTraining;
            isModelTraining = data.voice_id_training || data.emotion_training;
            
            // Обновляем интерфейс при изменении статуса
            if (isModelTraining !== wasTraining) {
                if (isModelTraining) {
                    showTrainingOverlay();
                } else {
                    hideTrainingOverlay();
                    
                    // После завершения обучения показываем уведомление
                    const message = "Обучение модели успешно завершено!";
                    showNotification(message, "success");
                }
            }
            
            // Обновляем информацию о тренировке
            updateTrainingInfo(data);
        })
        .catch(error => {
            // Не выводим ошибку в консоль, только логируем
            logErrorToSystem(error.message || "Ошибка проверки статуса", "status_checker");
        });
}

// Показать оверлей блокировки при обучении
function showTrainingOverlay() {
    let overlay = document.getElementById('training-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'training-overlay';
        overlay.className = 'training-overlay';
        
        const content = document.createElement('div');
        content.className = 'training-content';
        
        const title = document.createElement('h2');
        title.textContent = 'Идет обучение модели';
        
        const message = document.createElement('p');
        message.textContent = 'Пожалуйста, дождитесь завершения процесса обучения...';
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        
        const info = document.createElement('div');
        info.id = 'training-info';
        info.className = 'training-info';
        
        content.appendChild(title);
        content.appendChild(message);
        content.appendChild(spinner);
        content.appendChild(info);
        overlay.appendChild(content);
        
        document.body.appendChild(overlay);
        
        // Блокируем все ссылки
        blockNavigation();
        
        // После создания оверлея добавляем прогресс-бар
        showProgressBar();
    } else {
        overlay.style.display = 'flex';
        // Показываем прогресс-бар для уже существующего оверлея
        showProgressBar();
    }
}

// Скрыть оверлей блокировки
function hideTrainingOverlay() {
    const overlay = document.getElementById('training-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    
    // Останавливаем проверку прогресса
    if (intervals.progressCheck) {
        clearInterval(intervals.progressCheck);
        intervals.progressCheck = null;
    }
    
    // Сбрасываем тип обучения
    currentTrainingType = null;
    
    // Разблокируем навигацию
    unblockNavigation();
}

// Обновить информацию о тренирующейся модели
function updateTrainingInfo(data) {
    const infoElement = document.getElementById('training-info');
    if (infoElement) {
        if (data.voice_id_training) {
            infoElement.textContent = 'Обучается модель идентификации по голосу';
        } else if (data.emotion_training) {
            infoElement.textContent = 'Обучается модель распознавания эмоций';
        }
    }
}

// Блокировка навигации
function blockNavigation() {
    document.querySelectorAll('a').forEach(link => {
        link.dataset.originalHref = link.href;
        link.dataset.originalClick = link.onclick;
        
        link.onclick = function(e) {
            e.preventDefault();
            alert('Навигация заблокирована во время обучения модели. Пожалуйста, дождитесь завершения процесса.');
            return false;
        };
    });
}

// Разблокировка навигации
function unblockNavigation() {
    document.querySelectorAll('a').forEach(link => {
        if (link.dataset.originalHref) {
            link.href = link.dataset.originalHref;
        }
        if (link.dataset.originalClick) {
            link.onclick = link.dataset.originalClick;
        } else {
            link.onclick = null;
        }
    });
}

// Функция для инициализации всех интервалов
function initIntervals() {
    // Очищаем существующие интервалы перед созданием новых
    clearAllIntervals();
    
    // Запускаем проверку статуса моделей
    intervals.statusCheck = setInterval(checkModelStatus, 5000);
}

// Функция для очистки всех интервалов
function clearAllIntervals() {
    // Очищаем интервал проверки статуса
    if (intervals.statusCheck) {
        clearInterval(intervals.statusCheck);
        intervals.statusCheck = null;
    }
    
    // Очищаем интервал проверки прогресса
    if (intervals.progressCheck) {
        clearInterval(intervals.progressCheck);
        intervals.progressCheck = null;
    }
}

// Добавляем стили для оверлея только если их еще нет
document.addEventListener('DOMContentLoaded', () => {
    if (!checkBrowserCompatibility()) {
        return; // Останавливаем инициализацию при несовместимости
    }
    
    if (!document.getElementById('common-overlay-styles')) {
        const style = document.createElement('style');
        style.id = 'common-overlay-styles';
        style.textContent = `
            .training-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
            }
            
            .training-content {
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                text-align: center;
            }
            
            .spinner {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .training-info {
                margin-top: 15px;
                font-style: italic;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Инициализируем интервалы
    initIntervals();
    
    // Добавляем обработчик для очистки интервалов при уходе со страницы
    window.addEventListener('beforeunload', clearAllIntervals);
});

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

// Функция для отображения уведомлений
function showNotification(message, type = "info") {
    // Проверяем, существует ли уже контейнер для уведомлений
    let notificationContainer = document.getElementById('notification-container');
    
    if (!notificationContainer) {
        // Создаем контейнер для уведомлений
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notification-container';
        notificationContainer.style.position = 'fixed';
        notificationContainer.style.top = '20px';
        notificationContainer.style.right = '20px';
        notificationContainer.style.zIndex = '9999';
        document.body.appendChild(notificationContainer);
    }
    
    // Создаем уведомление
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = message;
    
    // Добавляем стили
    notification.style.backgroundColor = type === 'success' ? '#4CAF50' : 
                                        type === 'error' ? '#f44336' : 
                                        '#2196F3';
    notification.style.color = 'white';
    notification.style.padding = '15px 20px';
    notification.style.marginBottom = '10px';
    notification.style.borderRadius = '5px';
    notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    notification.style.opacity = '0';
    notification.style.transition = 'opacity 0.3s ease';
    
    // Добавляем уведомление в контейнер
    notificationContainer.appendChild(notification);
    
    // Отображаем уведомление с анимацией
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 10);
    
    // Скрываем и удаляем уведомление через 5 секунд
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.addEventListener('transitionend', function() {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }, 5000);
}

// Определяем, поддерживает ли браузер необходимые функции
function checkBrowserCompatibility() {
    const features = {
        fetch: typeof fetch !== 'undefined',
        promises: typeof Promise !== 'undefined',
        audioApi: typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined'
    };
    
    // Проверяем совместимость
    const incompatibleFeatures = Object.keys(features).filter(key => !features[key]);
    
    if (incompatibleFeatures.length > 0) {
        const message = `Ваш браузер не поддерживает следующие необходимые функции: ${incompatibleFeatures.join(', ')}. Пожалуйста, обновите браузер или используйте другой.`;
        alert(message);
        return false;
    }
    
    return true;
}

// Функция для отображения прогресс-бара
function showProgressBar() {
    let overlay = document.getElementById('training-overlay');
    if (!overlay) {
        return; // Если оверлея нет, значит не в режиме обучения
    }
    
    // Проверяем, есть ли уже прогресс-бар
    let progressContainer = document.getElementById('training-progress-container');
    if (!progressContainer) {
        const content = overlay.querySelector('.training-content');
        
        // Создаем контейнер для прогресс-бара
        progressContainer = document.createElement('div');
        progressContainer.id = 'training-progress-container';
        progressContainer.className = 'training-progress-container';
        
        // Создаем строку со статистикой
        const statsEl = document.createElement('div');
        statsEl.id = 'training-stats';
        statsEl.className = 'training-stats';
        statsEl.innerHTML = 'Подготовка данных...';
        
        // Создаем прогресс-бар
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar-container';
        
        const progressFill = document.createElement('div');
        progressFill.id = 'progress-fill';
        progressFill.className = 'progress-fill';
        progressFill.style.width = '0%';
        
        const progressText = document.createElement('div');
        progressText.id = 'progress-text';
        progressText.className = 'progress-text';
        progressText.textContent = '0%';
        
        // Собираем структуру прогресс-бара
        progressBar.appendChild(progressFill);
        progressBar.appendChild(progressText);
        
        // Добавляем элементы в контейнер
        progressContainer.appendChild(statsEl);
        progressContainer.appendChild(progressBar);
        
        // Добавляем контейнер в оверлей
        content.appendChild(progressContainer);
        
        // Начинаем проверку прогресса
        startProgressCheck();
    }
}

// Функция для запуска проверки прогресса
function startProgressCheck() {
    // Останавливаем предыдущие проверки, если есть
    if (intervals.progressCheck) {
        clearInterval(intervals.progressCheck);
    }
    
    // Запускаем новый интервал
    intervals.progressCheck = setInterval(checkTrainingProgress, 1000);
}

// Функция для проверки прогресса обучения
function checkTrainingProgress() {
    if (!currentTrainingType) {
        // Определяем тип обучения на основе URL
        if (window.location.pathname.includes('idtraining')) {
            currentTrainingType = 'voice_id';
        } else if (window.location.pathname.includes('emtraining')) {
            currentTrainingType = 'emotion';
        } else {
            currentTrainingType = 'all'; // Значение по умолчанию
        }
    }
    
    fetch(`/api/training_progress?model_type=${currentTrainingType}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateProgressBar(data);
        })
        .catch(error => {
            // Просто логируем ошибку, не показываем пользователю
            logErrorToSystem(error.message || "Ошибка проверки прогресса", "progress_checker");
        });
}

// Функция для обновления прогресс-бара
function updateProgressBar(data) {
    // Если получены данные для всех моделей, берем правильную
    const progress = currentTrainingType === 'all' ? 
                     (data.voice_id.status === 'training' ? data.voice_id : data.emotion) : 
                     data;
    
    // Если модель не обучается, скрываем прогресс-бар
    if (progress.status !== 'training') {
        if (intervals.progressCheck) {
            clearInterval(intervals.progressCheck);
            intervals.progressCheck = null;
        }
        return;
    }
    
    // Обновляем элементы прогресс-бара
    const statsEl = document.getElementById('training-stats');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    if (statsEl && progressFill && progressText) {
        // Вычисляем процент завершения
        const percent = Math.round((progress.current_epoch / progress.total_epochs) * 100);
        
        // Обновляем элементы
        progressFill.style.width = `${percent}%`;
        progressText.textContent = `${percent}%`;
        
        // Вычисляем прошедшее время
        const elapsedTime = Math.round((Date.now() - progress.start_time * 1000) / 1000);
        const minutes = Math.floor(elapsedTime / 60);
        const seconds = elapsedTime % 60;
        
        // Обновляем статистику
        statsEl.innerHTML = `
            Эпоха: ${progress.current_epoch} / ${progress.total_epochs}<br>
            Точность: ${(progress.accuracy * 100).toFixed(2)}%<br>
            Ошибка: ${progress.loss.toFixed(4)}<br>
            Прошло времени: ${minutes}м ${seconds}с
        `;
    }
}
