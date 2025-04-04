// Анимация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    const navigation = document.querySelector('.navigation');
    const description = document.querySelector('.description');
    
    // Добавляем небольшую задержку для анимации
    setTimeout(() => {
        navigation.style.opacity = '1';
    }, 200);
    
    setTimeout(() => {
        description.style.opacity = '1';
    }, 400);
});

// Анимация кнопок при наведении
const buttons = document.querySelectorAll('.nav-button');
buttons.forEach(button => {
    button.addEventListener('mouseover', () => {
        button.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.2)';
    });
    
    button.addEventListener('mouseout', () => {
        button.style.boxShadow = 'none';
    });
});
