"""Market environment - live or simulated market state."""

from src.core.logger import get_logger


class MarketEnvironment:
    """Represents market state and provides price/order book data."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.logger = get_logger("environment.market")

    def get_state(self) -> dict:
        """Return current market state."""
        return {"prices": {}, "order_book": {}}

    def step(self, action: dict) -> tuple[dict, float, bool]:
        """Execute action and return (next_state, reward, done)."""
        self.logger.debug("Market step: %s", action)
        return self.get_state(), 0.0, False
