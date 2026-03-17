"""ETL pipeline - extract, transform, load."""

from src.core.logger import get_logger


def run_etl(
    raw_path: str = "data/raw",
    processed_path: str = "data/processed",
    **kwargs,
) -> str:
    """Run ETL pipeline and return path to processed data."""
    logger = get_logger("pipelines.etl")
    logger.info("Running ETL: %s -> %s", raw_path, processed_path)
    return processed_path