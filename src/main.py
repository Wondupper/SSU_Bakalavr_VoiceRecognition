from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pathlib import Path
from backend.api.routes import router as api_router

app = FastAPI(title="Voice Recognition System")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка шаблонов
templates = Jinja2Templates(directory=Path("frontend/templates"))

# Монтирование статических файлов для каждой страницы
app.mount("/home/static", StaticFiles(directory=Path("frontend/templates/home/static")), name="home_static")
app.mount("/training/static", StaticFiles(directory=Path("frontend/templates/training/static")), name="training_static")
app.mount("/identification/static", StaticFiles(directory=Path("frontend/templates/identification/static")), name="identification_static")

# Подключение API роутов
app.include_router(api_router, prefix="/api")

# Маршруты для страниц
@app.get("/")
async def home():
    return templates.TemplateResponse("home/index.html", {"request": {}})

@app.get("/training")
async def training():
    return templates.TemplateResponse("training/index.html", {"request": {}})

@app.get("/identification")
async def identification():
    return templates.TemplateResponse("identification/index.html", {"request": {}})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 