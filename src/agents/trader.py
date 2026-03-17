"""Trader agent - synthesizes bullish and bearish reports."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.pipeline.config import PipelineConfig

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


def _build_input(bullish_report: str, bearish_report: str) -> str:
    return "\n\n".join(
        [
            "BULLISH ANALYST REPORT:\n" + (bullish_report or "(none)"),
            "BEARISH ANALYST REPORT:\n" + (bearish_report or "(none)"),
        ]
    )


async def trader_agent(state: AgentState, cfg: PipelineConfig) -> AgentState:
    """Create a trader recommendation based on analyst reports."""
    await asyncio.sleep(cfg.llm_request_delay_s)

    llm = get_llm(cfg)
    input_text = _build_input(state.get("bullish_report", ""), state.get("bearish_report", ""))

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=input_text)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        return {"trader_report": content.strip()}
    except Exception as exc:  # pragma: no cover - network/provider dependent
        return {
            "trader_report": "Unable to generate trader report.",
            "error": str(exc),
        }
