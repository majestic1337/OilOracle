"""Pipeline configuration for API + agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

from src.core.exceptions import ConfigurationError

load_dotenv()

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DB_URL = "postgresql://admin:adminpassword@localhost:5432/brentprices_data"
DEFAULT_LLM_PROVIDER = "google"
DEFAULT_LLM_MODEL = "gemini-flash-2.0"
DEFAULT_LLM_REQUEST_DELAY_S = 0.3
DEFAULT_DECISIONS_CACHE_DAYS = 30
DEFAULT_INITIAL_CAPITAL = 100000.0


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid integer for {name}: {value}") from exc


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid float for {name}: {value}") from exc


@dataclass(slots=True)
class PipelineConfig:
    """Central configuration for API endpoints and the agent team."""

    api_base_url: str = DEFAULT_API_BASE_URL
    decisions_cache_days: int = DEFAULT_DECISIONS_CACHE_DAYS
    db_url: str = DEFAULT_DB_URL

    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_request_delay_s: float = DEFAULT_LLM_REQUEST_DELAY_S

    initial_capital: float = DEFAULT_INITIAL_CAPITAL

    def __post_init__(self) -> None:
        self.api_base_url = _env("API_BASE_URL") or self.api_base_url
        self.decisions_cache_days = _env_int(
            "DECISIONS_CACHE_DAYS",
            self.decisions_cache_days,
        )
        self.db_url = _env("DB_URL") or self.db_url

        self.llm_provider = (_env("LLM_PROVIDER") or self.llm_provider).lower()
        self.llm_model = _env("LLM_MODEL") or self.llm_model
        self.llm_api_key = _env("LLM_API_KEY") or self.llm_api_key
        self.llm_base_url = _env("LLM_BASE_URL") or self.llm_base_url
        self.llm_request_delay_s = _env_float(
            "LLM_REQUEST_DELAY_S",
            self.llm_request_delay_s,
        )

        self.initial_capital = _env_float("INITIAL_CAPITAL", self.initial_capital)

        if not self.llm_base_url and self.llm_provider in ("ollama", "lmstudio"):
            self.llm_base_url = (
                "http://localhost:11434/v1"
                if self.llm_provider == "ollama"
                else "http://localhost:1234/v1"
            )

        if not self.api_base_url:
            raise ConfigurationError("api_base_url must not be empty.")
        if self.decisions_cache_days < 0:
            raise ConfigurationError("decisions_cache_days must be >= 0.")
        if self.initial_capital <= 0:
            raise ConfigurationError("initial_capital must be > 0.")


__all__ = ["PipelineConfig"]