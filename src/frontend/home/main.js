// Анимация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    // Убедимся, что мы находимся на главной странице
    if (window.location.pathname !== '/') return;
    
    // Анимация заголовка и описания на главной странице
    const heroTitle = document.querySelector('.hero-title');
    const heroSubtitle = document.querySelector('.hero-subtitle');
    const featureGrid = document.querySelector('.feature-grid');
    
    if (heroTitle) {
        setTimeout(() => {
            heroTitle.style.opacity = '1';
            heroTitle.style.transform = 'translateY(0)';
        }, 200);
    }
    
    if (heroSubtitle) {
        setTimeout(() => {
            heroSubtitle.style.opacity = '1';
            heroSubtitle.style.transform = 'translateY(0)';
        }, 400);
    }
    
    if (featureGrid) {
        setTimeout(() => {
            featureGrid.style.opacity = '1';
        }, 600);
    }
    
    // Анимация карточек при наведении
    const featureCards = document.querySelectorAll('.feature-card');
    if (featureCards && featureCards.length > 0) {
        featureCards.forEach(card => {
            card.addEventListener('mouseover', () => {
                card.style.transform = 'translateY(-5px)';
                card.style.boxShadow = '0 8px 20px rgba(0, 0, 0, 0.2)';
            });
            
            card.addEventListener('mouseout', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.1)';
            });
        });
    }
});

// Анимация кнопок при наведении - перемещаем внутрь DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    // Анимация кнопок при наведении - только если они существуют
    const buttons = document.querySelectorAll('.nav-button');
    if (buttons && buttons.length > 0) {
        buttons.forEach(button => {
            button.addEventListener('mouseover', () => {
                button.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.2)';
            });
            
            button.addEventListener('mouseout', () => {
                button.style.boxShadow = 'none';
            });
        });
    }
});
