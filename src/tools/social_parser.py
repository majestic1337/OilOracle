"""Social media parsing and sentiment extraction."""

from src.core.logger import get_logger


def parse_social(platform: str | None = None, query: str | None = None) -> list[dict]:
    """Parse social media posts and return structured items."""
    logger = get_logger("tools.social_parser")
    logger.info("Parsing social: platform=%s, query=%s", platform, query)
    return []
