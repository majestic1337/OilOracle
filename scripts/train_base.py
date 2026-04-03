"""Shared utilities for model-specific training scripts."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.pipelines.wfv_orchestrator import WFVConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CONFIG_CANDIDATES: tuple[Path, ...] = (
    PROJECT_ROOT / "configs" / "wfv_config.json",
    PROJECT_ROOT / "configs" / "config.yaml",
)
WFV_FIELD_NAMES = {field.name for field in fields(WFVConfig)}


def resolve_path(path: str | Path) -> Path:
    """Resolve relative paths from project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve and validate processed-data directory."""
    resolved = resolve_path(data_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Data directory not found: {resolved}")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"YAML config requested but PyYAML is unavailable: {path}") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def load_runtime_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load runtime config from explicit path or known defaults."""
    if config_path is not None:
        explicit = resolve_path(config_path)
        if not explicit.exists():
            raise FileNotFoundError(f"Config file not found: {explicit}")
        logger.info("Using config file: {path}", path=explicit)
        suffix = explicit.suffix.lower()
        if suffix == ".json":
            return _load_json(explicit)
        if suffix in {".yaml", ".yml"}:
            return _load_yaml(explicit)
        raise ValueError(f"Unsupported config format: {explicit}")

    candidates = [path for path in DEFAULT_CONFIG_CANDIDATES if path.exists()]
    if not candidates:
        logger.warning("No config file found; using WFV defaults and CLI overrides")
        return {}

    for path in candidates:
        try:
            logger.info("Trying config file: {path}", path=path)
            suffix = path.suffix.lower()
            if suffix == ".json":
                return _load_json(path)
            if suffix in {".yaml", ".yml"}:
                return _load_yaml(path)
            logger.warning("Skipping unsupported config format: {path}", path=path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load config {path}: {error}", path=path, error=str(exc))

    logger.warning("No readable config file found; using WFV defaults and CLI overrides")
    return {}


def get_model_config(config_payload: dict[str, Any], model_key: str) -> dict[str, Any]:
    """Extract optional model-specific section from config payload."""
    if not config_payload:
        return {}

    direct = config_payload.get(model_key)
    if isinstance(direct, dict):
        return direct

    models = config_payload.get("models")
    if isinstance(models, dict):
        candidate = models.get(model_key)
        if isinstance(candidate, dict):
            return candidate

    return {}


def _extract_wfv_config(config_payload: dict[str, Any]) -> dict[str, Any]:
    wfv_settings: dict[str, Any] = {}

    for key in ("wfv", "wfv_config"):
        section = config_payload.get(key)
        if isinstance(section, dict):
            wfv_settings.update(section)

    for key, value in config_payload.items():
        if key in WFV_FIELD_NAMES and key not in wfv_settings:
            wfv_settings[key] = value

    return wfv_settings


def build_wfv_config(
    config_payload: dict[str, Any],
    horizon: int,
    model_family: str,
) -> WFVConfig:
    """Build WFVConfig from config payload + CLI overrides."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    params = {
        key: value
        for key, value in _extract_wfv_config(config_payload).items()
        if key in WFV_FIELD_NAMES
    }
    params["horizon"] = horizon
    params["model_family"] = model_family
    return WFVConfig(**params)


def load_ml_data(data_dir: Path, horizon: int) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    """Load ML feature matrix and shifted target."""
    feature_path = data_dir / "feature_matrix_ml.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"ML feature matrix not found: {feature_path}")

    target_path = data_dir / f"target_h{horizon}.parquet"
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    X = pd.read_parquet(feature_path)
    y = pd.read_parquet(target_path)
    y_obj: pd.Series | pd.DataFrame = y.iloc[:, 0] if y.shape[1] == 1 else y

    common_index = X.index.intersection(y_obj.index)
    return X.loc[common_index].sort_index(), y_obj.loc[common_index].sort_index()


def load_stat_data(data_dir: Path, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Load stat-model features with properly shifted target matching ML semantics."""
    dl_path = data_dir / "feature_matrix_dl.parquet"
    if not dl_path.exists():
        raise FileNotFoundError(f"DL feature matrix (used for stat baselines) not found: {dl_path}")

    target_path = data_dir / f"target_h{horizon}.parquet"
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    X_full = pd.read_parquet(dl_path)
    if "ds" in X_full.columns:
        X_full["ds"] = pd.to_datetime(X_full["ds"])
        X_full = X_full.set_index("ds")
    
    # Видалення колонки 'y' (незсунутої), якщо вона є, щоб уникнути витоку
    if "y" in X_full.columns:
        X_full = X_full.drop(columns=["y"])
    if "unique_id" in X_full.columns:
        X_full = X_full.drop(columns=["unique_id"])

    y_shifted = pd.read_parquet(target_path)
    y_obj = y_shifted.iloc[:, 0]

    common_index = X_full.index.intersection(y_obj.index)
    return X_full.loc[common_index].sort_index(), y_obj.loc[common_index].sort_index()


def load_dl_data(data_dir: Path, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Load DL data in log-return space aligned with ML semantics.

    Uses:
    - `feature_matrix_ml.parquet` as exogenous predictors
    - `target_h{horizon}.parquet` as shifted target in return space

    Returned X contains NeuralForecast-compatible metadata columns
    (`unique_id`, `ds`) plus exogenous predictors. Target y is the shifted
    return series named `y`.
    """
    feature_path = data_dir / "feature_matrix_ml.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"ML feature matrix not found for DL training: {feature_path}")

    target_path = data_dir / f"target_h{horizon}.parquet"
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    X_ml = pd.read_parquet(feature_path)
    y_shifted = pd.read_parquet(target_path)
    y_obj = y_shifted.iloc[:, 0]

    common_index = X_ml.index.intersection(y_obj.index)
    X_aligned = X_ml.loc[common_index].sort_index()
    y_aligned = y_obj.loc[common_index].sort_index().astype(float).rename("y")

    X_dl = X_aligned.copy()
    X_dl.insert(0, "unique_id", "brent")
    X_dl.insert(1, "ds", pd.DatetimeIndex(X_dl.index))

    return X_dl, y_aligned
