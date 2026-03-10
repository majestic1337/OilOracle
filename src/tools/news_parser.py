"""News parsing and sentiment extraction."""

from src.core.logger import get_logger


def parse_news(source: str | None = None, limit: int = 100) -> list[dict]:
    """Parse news from source and return structured items."""
    logger = get_logger("tools.news_parser")
    logger.info("Parsing news from %s", source or "default")
    return []
