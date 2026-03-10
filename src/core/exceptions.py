"""Custom exceptions."""


class BaseAppError(Exception):
    """Base exception for application errors."""

    pass


class ConfigurationError(BaseAppError):
    """Raised when configuration is invalid or missing."""

    pass


class DataError(BaseAppError):
    """Raised when data processing fails."""

    pass


class ModelError(BaseAppError):
    """Raised when model inference or training fails."""

    pass


class TradingError(BaseAppError):
    """Raised when trading operations fail."""

    pass
