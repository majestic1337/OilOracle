"""Risk agent - risk assessment and position sizing."""

from src.agents.base import BaseAgent
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.pipelines.config import PipelineConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.agents.graph import AgentState

SYSTEM_PROMPT = (
    "You are a risk manager for an oil trading desk.\n"
    "Evaluate current market risk based on quantitative indicators and recent decision history.\n"
    "Be concise and quantitative.\n"
    "Your response must include these exact lines:\n"
    "RISK_LEVEL: LOW | MEDIUM | HIGH\n"
    "ADJUSTED_SIZE: <float between 0.0 and 1.0>\n"
    "WARNING: <one line or \"none\">"
)

RISK_KEYS = [
    "suggested_position_size",
    "stop_loss_atr",
    "risk_reward_ratio",
    "volatility_regime",
    "signal_strength",
    "model_agreement",
]


class RiskAgent(BaseAgent):
    """Agent responsible for risk assessment and position sizing."""

    def __init__(self):
        super().__init__("risk")

    def run(self, analysis: dict | None = None, **kwargs) -> dict:
        """Assess risk and compute position sizes."""
        self.logger.info("Running risk agent")
        return {"risk_score": 0.0, "position_sizes": {}}
