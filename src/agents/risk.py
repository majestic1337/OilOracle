"""Risk agent - risk assessment and position sizing."""

from src.agents.base import BaseAgent


class RiskAgent(BaseAgent):
    """Agent responsible for risk assessment and position sizing."""

    def __init__(self):
        super().__init__("risk")

    def run(self, analysis: dict | None = None, **kwargs) -> dict:
        """Assess risk and compute position sizes."""
        self.logger.info("Running risk agent")
        return {"risk_score": 0.0, "position_sizes": {}}
