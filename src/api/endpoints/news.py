"""News endpoints."""

from fastapi import APIRouter

router = APIRouter()

MOCK_NEWS = {
    "news": [
        {
            "date": "2024-06-15",
            "headline": "OPEC+ maintains output cuts through Q3",
            "category": "OPEC",
            "sentiment": "bullish",
        },
        {
            "date": "2024-06-14",
            "headline": "US dollar weakens on Fed pause signals",
            "category": "macro",
            "sentiment": "bullish",
        },
        {
            "date": "2024-06-13",
            "headline": "China demand outlook revised downward",
            "category": "demand",
            "sentiment": "bearish",
        },
    ]
}


@router.get("/news/recent")
def recent_news() -> dict:
    """Return mock news items."""
    return MOCK_NEWS
