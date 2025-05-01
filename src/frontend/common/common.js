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
    
    // Создаем элемент уведомления
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.backgroundColor = type === 'success' ? '#4CAF50' : 
                                         type === 'error' ? '#f44336' : '#2196F3';
    notification.style.color = 'white';
    notification.style.padding = '15px';
    notification.style.marginBottom = '10px';
    notification.style.borderRadius = '4px';
    notification.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.2)';
    notification.style.opacity = '0';
    notification.style.transition = 'opacity 0.3s';
    
    // Создаем текст уведомления
    notification.textContent = message;
    
    // Добавляем кнопку закрытия
    const closeButton = document.createElement('span');
    closeButton.textContent = '×';
    closeButton.style.float = 'right';
    closeButton.style.cursor = 'pointer';
    closeButton.style.fontWeight = 'bold';
    closeButton.style.marginLeft = '10px';
    closeButton.onclick = function() {
        notification.style.opacity = '0';
        setTimeout(() => notificationContainer.removeChild(notification), 300);
    };
    
    notification.insertBefore(closeButton, notification.firstChild);
    
    // Добавляем уведомление в контейнер
    notificationContainer.appendChild(notification);
    
    // Отображаем уведомление с анимацией
    setTimeout(() => notification.style.opacity = '1', 10);
    
    // Устанавливаем таймер для автоматического закрытия уведомления
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentNode) {
                    notificationContainer.removeChild(notification);
                }
            }, 300);
        }
    }, 5000);
}

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

// Добавляем инициализацию при загрузке DOM
document.addEventListener('DOMContentLoaded', () => {
    // Проверяем совместимость браузера
    if (!checkBrowserCompatibility()) {
        return; // Останавливаем инициализацию при несовместимости
    }
});
