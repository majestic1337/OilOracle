"""ML model training."""

from src.core.logger import get_logger


def train_model(data_path: str, config: dict | None = None) -> str:
    """Train model and return path to saved artifact."""
    logger = get_logger("models.ml.train")
    logger.info("Training model on %s", data_path)
    return "models/artifact.pt"
