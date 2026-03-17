"""Manager agent - orchestrates other agents."""

from src.agents.analyst import AnalystAgent
from src.agents.base import BaseAgent
from src.agents.risk import RiskAgent
from src.agents.trader import TraderAgent
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.pipelines.config import PipelineConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.agents.graph import AgentState

SYSTEM_PROMPT = (
    "You are the portfolio manager making the final trading decision.\n"
    "You receive a trader recommendation and a risk assessment.\n"
    "Your response must be valid JSON only — no markdown, no explanation outside JSON.\n"
    "JSON schema:\n"
    "{\n"
    "  \"decision\": \"BUY\" | \"SELL\" | \"HOLD\",\n"
    "  \"direction\": \"UP\" | \"DOWN\" | \"NEUTRAL\",\n"
    "  \"position_size\": <float 0.0-1.0>,\n"
    "  \"stop_loss\": <float, price level>,\n"
    "  \"take_profit\": <float, price level>,\n"
    "  \"rationale\": \"<two sentences max>\",\n"
    "  \"timestamp\": \"<ISO 8601>\"\n"
    "}"
)


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
