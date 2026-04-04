"""
Analysis route handlers — stock screening and market regime detection.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException

from backtester.screener import screen_universe
from backtester.regime import detect_regime
from api.models import (
    ScreenRequest,
    ScreenResponse,
    ScreenerScore,
    ScreenerError,
    RegimeRequest,
    RegimeResponse,
    RegimeIndicatorModel,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

_EXECUTOR = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))


# ---------------------------------------------------------------------------
# Stock screener
# ---------------------------------------------------------------------------

def _run_screen_sync(req: ScreenRequest) -> dict:
    results, errors = screen_universe(
        tickers=req.tickers,
        benchmark=req.benchmark,
        start=req.start_date,
        end=req.end_date,
        top_n=req.top_n,
    )

    # Determine actual date range used
    end_date = req.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = req.start_date or (datetime.now() - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )

    return {
        "benchmark": req.benchmark,
        "start_date": start_date,
        "end_date": end_date,
        "total_screened": len(results) + len(errors),
        "results": [
            ScreenerScore(
                ticker=r.ticker,
                rank=r.rank,
                composite_score=r.composite_score,
                relative_strength=r.relative_strength,
                momentum_1m=r.momentum_1m,
                momentum_3m=r.momentum_3m,
                momentum_6m=r.momentum_6m,
                momentum_12m=r.momentum_12m,
                sharpe=r.sharpe,
                volatility=r.volatility,
                max_drawdown=r.max_drawdown,
            )
            for r in results
        ],
        "errors": [ScreenerError(ticker=e["ticker"], error=e["error"]) for e in errors],
    }


@router.post("/screen", response_model=ScreenResponse)
async def screen_stocks(req: ScreenRequest) -> ScreenResponse:
    """Screen and rank stocks relative to the market."""
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(_EXECUTOR, _run_screen_sync, req)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ScreenResponse(**data)


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def _run_regime_sync(req: RegimeRequest) -> dict:
    result = detect_regime(
        benchmark=req.benchmark,
        start=req.start_date,
        end=req.end_date,
        bullish_threshold=req.bullish_threshold,
        bearish_threshold=req.bearish_threshold,
    )
    return {
        "regime": result.regime.value,
        "composite_score": result.composite_score,
        "confidence": result.confidence,
        "as_of_date": result.as_of_date,
        "indicators": [
            RegimeIndicatorModel(
                name=ind.name,
                value=ind.value,
                score=ind.score,
                interpretation=ind.interpretation,
            )
            for ind in result.indicators
        ],
    }


@router.post("/regime", response_model=RegimeResponse)
async def market_regime(req: RegimeRequest) -> RegimeResponse:
    """Detect the current market regime (bullish / bearish / neutral)."""
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(_EXECUTOR, _run_regime_sync, req)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return RegimeResponse(**data)
