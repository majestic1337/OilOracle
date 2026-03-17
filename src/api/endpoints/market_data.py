"""Market diagnostic endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

MOCK_DIAGNOSTIC = {
    "as_of_date": "2024-06-15",
    "current_price": 85.4,
    "price_change_1d": 2.3,
    "price_change_5d": -1.1,
    "trend_direction": "above_sma50",
    "confidence_score": 0.68,
    "signal_strength": 0.18,
    "signal_direction": "UP",
    "model_agreement": 0.92,
    "recent_model_accuracy": 0.61,
    "dxy_change_1d": -0.3,
    "gold_change_1d": 0.8,
    "brent_wti_spread": 3.2,
    "volatility_regime": "high",
    "trend_strength": 28.4,
    "market_regime": "bull",
    "suggested_position_size": 0.6,
    "stop_loss_atr": 1.5,
    "take_profit_atr": 2.5,
    "risk_reward_ratio": 1.67,
}


@router.get("/market/diagnostic")
def market_diagnostic() -> dict:
    """Return mock diagnostic payload; never raises."""
    try:
        return MOCK_DIAGNOSTIC
    except Exception:  # pragma: no cover - defensive safeguard
        logger.exception("Market diagnostic failed; returning mock payload")
        return MOCK_DIAGNOSTIC
