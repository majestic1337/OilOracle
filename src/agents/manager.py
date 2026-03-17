"""Manager agent - final decision synthesis."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.pipeline.config import PipelineConfig

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


def parse_manager_output(raw: str) -> Dict[str, Any] | None:
    """Parse manager JSON output, with markdown fallback."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def _safe_default() -> Dict[str, Any]:
    return {
        "decision": "HOLD",
        "direction": "NEUTRAL",
        "position_size": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "rationale": "Parse error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required fields exist and timestamp is set."""
    normalized = dict(payload)
    if "timestamp" not in normalized or not normalized.get("timestamp"):
        normalized["timestamp"] = datetime.now(timezone.utc).isoformat()
    return normalized


def _build_input(trader_report: str, risk_report: str) -> str:
    return "\n\n".join(
        [
            "TRADER RECOMMENDATION:\n" + (trader_report or "(none)"),
            "RISK ASSESSMENT:\n" + (risk_report or "(none)"),
        ]
    )


async def manager_agent(state: AgentState, cfg: PipelineConfig) -> AgentState:
    """Create the final manager decision JSON."""
    await asyncio.sleep(cfg.llm_request_delay_s)

    llm = get_llm(cfg)
    input_text = _build_input(state.get("trader_report", ""), state.get("risk_report", ""))

    parsed: Dict[str, Any] | None = None
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=input_text)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_manager_output(content)

        if parsed is None:
            retry_msg = "Return ONLY valid JSON. No markdown. No explanation."
            retry_response = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=input_text),
                    HumanMessage(content=retry_msg),
                ]
            )
            retry_content = (
                retry_response.content if hasattr(retry_response, "content") else str(retry_response)
            )
            parsed = parse_manager_output(retry_content)
    except Exception as exc:  # pragma: no cover - network/provider dependent
        return {"final_decision": _safe_default(), "error": str(exc)}

    if parsed is None:
        return {"final_decision": _safe_default()}
    return {"final_decision": _normalize_decision(parsed)}
