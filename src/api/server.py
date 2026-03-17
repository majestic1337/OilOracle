"""FastAPI server wiring for OilOracle."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import decisions, market_data, ml_signal, news

app = FastAPI(title="OilOracle API")

localhost_origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=localhost_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_data.router)
app.include_router(ml_signal.router)
app.include_router(news.router)
app.include_router(decisions.router)
