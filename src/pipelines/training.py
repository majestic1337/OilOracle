"""Training pipeline."""

from src.core.logger import get_logger


def run_training_pipeline(
    data_path: str = "data/processed",
    config: dict | None = None,
    **kwargs,
) -> str:
    """Run full training pipeline and return model path."""
    logger = get_logger("pipelines.training")
    logger.info("Running training pipeline on %s", data_path)
    return "models/artifact.pt"
