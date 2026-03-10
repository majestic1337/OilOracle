"""ML model inference."""

from src.core.logger import get_logger


def predict(features, model_path: str | None = None) -> list[float]:
    """Run inference on features."""
    logger = get_logger("models.ml.inference")
    logger.debug("Running inference")
    return []
