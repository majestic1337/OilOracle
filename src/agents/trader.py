"""Trader agent - order execution and trade management."""

from src.agents.base import BaseAgent


class TraderAgent(BaseAgent):
    """Agent responsible for executing trades based on signals."""

    def __init__(self):
        super().__init__("trader")

    def run(self, analysis: dict | None = None, risk: dict | None = None, **kwargs) -> list:
        """Generate and execute trade orders."""
        self.logger.info("Running trader agent")
        return []
