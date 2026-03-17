"""Bearish analyst agent."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.pipeline.config import PipelineConfig

if TYPE_CHECKING:  # pragma: no cover
    from src.agents.graph import AgentState

SYSTEM_PROMPT = (
    "You are a bearish oil market analyst specializing in Brent crude.\n"
    "Your job is to find and present the strongest arguments FOR a price decrease.\n"
    "Be specific. Reference the data provided. Do not mention bullish factors.\n"
    "Structure your response as numbered arguments.\n"
    "End with: SUMMARY: <one sentence confidence statement>"
)


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _format_diagnostic(diagnostic: Dict[str, Any]) -> str:
    if not diagnostic:
        return "No diagnostic data available."

    return "\n".join(
        [
            f"As of {_fmt_value(diagnostic.get('as_of_date'))}, Brent is {_fmt_value(diagnostic.get('current_price'))}.",
            "1D change: "
            f"{_fmt_value(diagnostic.get('price_change_1d'))}; "
            "5D change: "
            f"{_fmt_value(diagnostic.get('price_change_5d'))}.",
            "Trend: "
            f"{_fmt_value(diagnostic.get('trend_direction'))} with strength "
            f"{_fmt_value(diagnostic.get('trend_strength'))}.",
            "Signal: "
            f"{_fmt_value(diagnostic.get('signal_direction'))} with strength "
            f"{_fmt_value(diagnostic.get('signal_strength'))}; model agreement "
            f"{_fmt_value(diagnostic.get('model_agreement'))}.",
            "Market regime: "
            f"{_fmt_value(diagnostic.get('market_regime'))}; volatility "
            f"{_fmt_value(diagnostic.get('volatility_regime'))}.",
            "Macro: DXY 1D "
            f"{_fmt_value(diagnostic.get('dxy_change_1d'))}; Gold 1D "
            f"{_fmt_value(diagnostic.get('gold_change_1d'))}; Brent-WTI spread "
            f"{_fmt_value(diagnostic.get('brent_wti_spread'))}.",
        ]
    )


def _format_news(news: List[Dict[str, Any]]) -> str:
    if not news:
        return "No recent news available."
    lines = ["Recent news:"]
    for item in news:
        lines.append(
            f"- {item.get('date', 'n/a')}: {item.get('headline', 'n/a')} "
            f"({item.get('category', 'n/a')}, {item.get('sentiment', 'n/a')})"
        )
    return "\n".join(lines)


def _format_decisions(decisions: List[Dict[str, Any]]) -> str:
    if not decisions:
        return "No recent decisions available."
    lines = ["Recent decisions:"]
    for item in decisions[:5]:
        lines.append(
            f"- {item.get('timestamp', 'n/a')}: {item.get('decision', 'n/a')} "
            f"{item.get('direction', '')} (confidence {_fmt_value(item.get('confidence'))})"
        )
    return "\n".join(lines)


def _build_input(state: Dict[str, Any]) -> str:
    diagnostic = _format_diagnostic(state.get("diagnostic", {}))
    news = _format_news(state.get("news", []))
    decisions = _format_decisions(state.get("decisions_cache", []))
    return "\n\n".join(
        [
            "MARKET DIAGNOSTIC:\n" + diagnostic,
            "NEWS FLOW:\n" + news,
            "DECISION HISTORY:\n" + decisions,
        ]
    )


async def bearish_agent(state: AgentState, cfg: PipelineConfig) -> AgentState:
    """Generate the bearish analyst report."""
    await asyncio.sleep(cfg.llm_request_delay_s)

    llm = get_llm(cfg)
    input_text = _build_input(state)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=input_text)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        return {"bearish_report": content.strip()}
    except Exception as exc:  # pragma: no cover - network/provider dependent
        return {
            "bearish_report": "Unable to generate bearish report.",
            "error": str(exc),
        }
