"""Logging configuration."""

import logging
import sys

from src.core.constants import LOG_FORMAT, LOG_LEVEL


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level or LOG_LEVEL)
    return logger
