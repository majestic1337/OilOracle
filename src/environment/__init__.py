"""Trading environment components."""

from __future__ import annotations

from src.environment.simulator import PaperTradingSimulator

try:  # pragma: no cover - optional modules may not exist in all environments
    from src.environment.backtest import BacktestEnvironment
except Exception:  # pragma: no cover - defensive import
    BacktestEnvironment = None

try:  # pragma: no cover - optional modules may not exist in all environments
    from src.environment.market import MarketEnvironment
except Exception:  # pragma: no cover - defensive import
    MarketEnvironment = None

__all__ = ["MarketEnvironment", "BacktestEnvironment", "PaperTradingSimulator"]
