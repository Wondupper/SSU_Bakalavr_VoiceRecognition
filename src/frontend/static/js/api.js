/**
 * Базовый класс для работы с API
 */
export class API {
    static async makeRequest(url, method = 'GET', data = null) {
        try {
            const options = {
                method,
                headers: {}
            };

            if (data instanceof FormData) {
                options.body = data;
            } else if (data) {
                options.headers['Content-Type'] = 'application/json';
                options.body = JSON.stringify(data);
            }

            const response = await fetch(url, options);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Произошла ошибка при выполнении запроса');
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    /**
     * Отправка аудиофайлов и имени пользователя для обучения
     */
    static async trainModels(files, userName) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        formData.append('user_name', userName);

        return this.makeRequest('/api/train', 'POST', formData);
    }

    /**
     * Идентификация пользователя и проверка эмоции
     */
    static async identifyUser(file, targetEmotion) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_emotion', targetEmotion);

        return this.makeRequest('/api/identify', 'POST', formData);
    }

    /**
     * Получение статуса моделей
     */
    static async getModelsStatus() {
        return this.makeRequest('/api/status');
    }
}

/**
 * Класс для управления состоянием обработки
 */
export class ProcessingState {
    constructor() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'processing-overlay';
        this.overlay.innerHTML = '<div class="spinner"></div>';
    }

    show() {
        document.body.appendChild(this.overlay);
        document.body.style.overflow = 'hidden';
    }

    hide() {
        document.body.removeChild(this.overlay);
        document.body.style.overflow = '';
    }
}

/**
 * Класс для работы с уведомлениями
 */
export class Notifications {
    static show(message, type = 'success') {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} fade-in`;
        alert.textContent = message;

        document.body.appendChild(alert);

        setTimeout(() => {
            alert.remove();
        }, 3000);
    }

    static error(message) {
        this.show(message, 'error');
    }

    static success(message) {
        this.show(message, 'success');
    }
} 