"""Risk manager agent."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.pipeline.config import PipelineConfig

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


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _format_risk_fields(diagnostic: Dict[str, Any]) -> str:
    risk_fields = {key: diagnostic.get(key) for key in RISK_KEYS}
    lines = ["Risk indicators:"]
    for key, value in risk_fields.items():
        lines.append(f"- {key}: {_fmt_value(value)}")
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
    diagnostic = state.get("diagnostic", {})
    risk_text = _format_risk_fields(diagnostic)
    decisions_text = _format_decisions(state.get("decisions_cache", []))
    return "\n\n".join(
        [
            "RISK DATA:\n" + risk_text,
            "DECISION HISTORY:\n" + decisions_text,
        ]
    )


async def risk_agent(state: AgentState, cfg: PipelineConfig) -> AgentState:
    """Generate the risk manager report."""
    await asyncio.sleep(cfg.llm_request_delay_s)

    llm = get_llm(cfg)
    input_text = _build_input(state)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=input_text)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        return {"risk_report": content.strip()}
    except Exception as exc:  # pragma: no cover - network/provider dependent
        return {
            "risk_report": "Unable to generate risk report.",
            "error": str(exc),
        }
