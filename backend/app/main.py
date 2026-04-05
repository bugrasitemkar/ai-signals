from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, chat, signals

app = FastAPI(title="AI Signals Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(signals.router)
