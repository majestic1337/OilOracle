"""Backtest environment - historical simulation."""

from src.core.logger import get_logger


class BacktestEnvironment:
    """Backtesting environment using historical data."""

    def __init__(self, data_path: str | None = None, config: dict | None = None):
        self.data_path = data_path or "data/processed"
        self.config = config or {}
        self.logger = get_logger("environment.backtest")

    def run(self, agent, start_date: str | None = None, end_date: str | None = None) -> dict:
        """Run backtest with given agent and date range."""
        self.logger.info("Running backtest from %s to %s", start_date, end_date)
        return {"pnl": 0.0, "trades": [], "metrics": {}}
