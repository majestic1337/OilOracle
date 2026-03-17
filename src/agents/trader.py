"""Trader agent - order execution and trade management."""

<<<<<<< Updated upstream
from src.agents.base import BaseAgent
=======
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.pipelines.config import PipelineConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.agents.graph import AgentState

SYSTEM_PROMPT = (
    "You are an experienced oil market trader.\n"
    "You receive analysis from both a bullish and bearish analyst.\n"
    "Weigh both sides objectively. Make a clear directional decision.\n"
    "End your response with exactly one of:\n"
    "DECISION: BUY\n"
    "DECISION: SELL  \n"
    "DECISION: HOLD"
)
>>>>>>> Stashed changes


class TraderAgent(BaseAgent):
    """Agent responsible for executing trades based on signals."""

    def __init__(self):
        super().__init__("trader")

    def run(self, analysis: dict | None = None, risk: dict | None = None, **kwargs) -> list:
        """Generate and execute trade orders."""
        self.logger.info("Running trader agent")
        return []
