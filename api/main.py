"""
TDA Backtester — FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.backtest import router as backtest_router
from api.routes.analysis import router as analysis_router

app = FastAPI(
    title="TDA Backtester API",
    description=(
        "REST API for running algorithmic trading backtests. "
        "Supports SMA Crossover, Bollinger Mean Reversion, and Breakout Momentum strategies."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Vite dev server (port 5173) and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(backtest_router)
app.include_router(analysis_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
