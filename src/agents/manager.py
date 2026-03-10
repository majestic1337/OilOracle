"""Manager agent - orchestrates other agents."""

from src.agents.base import BaseAgent
from src.agents.analyst import AnalystAgent
from src.agents.risk import RiskAgent
from src.agents.trader import TraderAgent


class ManagerAgent(BaseAgent):
    """Orchestrates analyst, risk, and trader agents."""

    def __init__(self):
        super().__init__("manager")
        self.analyst = AnalystAgent()
        self.risk = RiskAgent()
        self.trader = TraderAgent()

    def run(self, market_data: dict | None = None, **kwargs) -> dict:
        """Coordinate agent workflow and return aggregated results."""
        self.logger.info("Running manager agent")
        analysis = self.analyst.run(market_data=market_data)
        risk_assessment = self.risk.run(analysis=analysis)
        trades = self.trader.run(analysis=analysis, risk=risk_assessment)
        return {"analysis": analysis, "risk": risk_assessment, "trades": trades}
