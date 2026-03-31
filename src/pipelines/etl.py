"""ETL pipeline - extract, transform, load."""

from pathlib import Path

from src.core.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _resolve_from_project_root(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def run_etl(
    raw_path: str | Path = DEFAULT_RAW_DIR,
    processed_path: str | Path = DEFAULT_PROCESSED_DIR,
    **kwargs,
) -> str:
    """Run ETL pipeline and return path to processed data."""
    del kwargs
    logger = get_logger("pipelines.etl")
    raw_dir = _resolve_from_project_root(raw_path)
    processed_dir = _resolve_from_project_root(processed_path)

    logger.info("Running ETL: %s -> %s", raw_dir, processed_dir)
    return str(processed_dir)
