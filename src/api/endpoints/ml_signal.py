"""ML signal endpoints."""

from fastapi import APIRouter

router = APIRouter()

MOCK_SIGNAL = {
    "confidence_score": 0.68,
    "signal_direction": "UP",
    "model_name": "XGBoost_cls",
    "as_of_date": "2024-06-15",
}


@router.get("/signal/latest")
def latest_signal() -> dict:
    """Return mock ML signal."""
    return MOCK_SIGNAL
