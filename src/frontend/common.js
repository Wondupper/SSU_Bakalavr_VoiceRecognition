// Глобальный статус для отслеживания обучения моделей
let isModelTraining = false;

// Проверка статуса моделей каждые 5 секунд
function checkModelStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            const wasTraining = isModelTraining;
            isModelTraining = data.voice_id_training || data.emotion_training;
            
            // Обновляем интерфейс при изменении статуса
            if (isModelTraining !== wasTraining) {
                if (isModelTraining) {
                    showTrainingOverlay();
                } else {
                    hideTrainingOverlay();
                }
            }
            
            // Обновляем информацию о тренировке
            updateTrainingInfo(data);
        })
        .catch(error => logErrorToSystem(error, "status_checker"));
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
    } else {
        overlay.style.display = 'flex';
    }
}

// Скрыть оверлей блокировки
function hideTrainingOverlay() {
    const overlay = document.getElementById('training-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    
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

// Добавляем стили для оверлея
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
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
    
    // Сохраняем идентификаторы интервалов для возможности их очистки
    let statusCheckIntervalId = null;
    let errorCheckIntervalId = null;
    
    // Функция для очистки интервалов при уходе со страницы
    const clearAllIntervals = () => {
        if (statusCheckIntervalId) clearInterval(statusCheckIntervalId);
        if (errorCheckIntervalId) clearInterval(errorCheckIntervalId);
    };
    
    // Добавляем обработчик, который очистит интервалы при уходе со страницы
    window.addEventListener('beforeunload', clearAllIntervals);
    
    // Запускаем периодическую проверку статуса
    statusCheckIntervalId = setInterval(checkModelStatus, 5000);
    
    // Добавляем проверку ошибок на странице админа
    if (window.location.pathname === '/panel') {
        const errorContainer = document.getElementById('system-errors');
        if (errorContainer) {
            // Запускаем периодическую проверку ошибок
            errorCheckIntervalId = setInterval(() => checkSystemErrors(errorContainer), 10000);
            
            // И проверяем сразу при загрузке страницы
            checkSystemErrors(errorContainer);
        }
    }
});

// Функция для проверки системных ошибок
function checkSystemErrors(displayElement = null) {
    // Запрашиваем последние ошибки от API
    fetch('/api/errors?limit=5')
        .then(response => response.json())
        .then(data => {
            const errors = data.errors;
            
            // Если есть ошибки и указан элемент для отображения
            if (errors.length > 0 && displayElement) {
                const errorsList = errors.map(error => {
                    const date = new Date(error.timestamp * 1000).toLocaleString();
                    return `<div class="error-item">
                        <span class="error-time">${date}</span>
                        <span class="error-module">${error.module || 'Система'}</span>: 
                        <span class="error-message">${error.message}</span>
                    </div>`;
                }).join('');
                
                displayElement.innerHTML = `<div class="system-errors">
                    <h4>Системные уведомления:</h4>
                    ${errorsList}
                </div>`;
                
                displayElement.style.display = 'block';
            }
        })
        .catch(error => logErrorToSystem(error, "system_errors"));
}

// Определяем глобальную функцию логирования ошибок
window.logErrorToSystem = function(error, module = "frontend", location = window.location.pathname) {
    // Локальное логирование для отладки (можно отключить в продакшн)
    console.error(`[${module}] ${error}`);
    
    // Попытка отправить ошибку на сервер, если это возможно
    try {
        fetch('/api/errors/log', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: error.toString(),
                module: module,
                location: location
            })
        }).catch(() => {
            // Игнорируем ошибки при отправке ошибок
        });
    } catch (e) {
        // Игнорируем ошибки при отправке ошибок
    }
};

// Глобальный обработчик ошибок
window.onerror = function(message, source, lineno, colno, error) {
    logErrorToSystem(message, "global", source + ":" + lineno);
    return false;
};
