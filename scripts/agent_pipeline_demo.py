"""Run a mock agent pipeline with a local llama3.2 model."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_llm
from src.agents.bearish import bearish_agent
from src.agents.bullish import bullish_agent
from src.agents.manager import manager_agent
from src.agents.risk import risk_agent
from src.agents.trader import trader_agent
from src.pipelines.config import PipelineConfig


MOCK_DIAGNOSTIC: Dict[str, Any] = {
    "as_of_date": "2024-06-15",
    "current_price": 85.4,
    "price_change_1d": 2.3,
    "price_change_5d": -1.1,
    "trend_direction": "above_sma50",
    "confidence_score": 0.68,
    "signal_strength": 0.18,
    "signal_direction": "UP",
    "model_agreement": 0.92,
    "recent_model_accuracy": 0.61,
    "dxy_change_1d": -0.3,
    "gold_change_1d": 0.8,
    "brent_wti_spread": 3.2,
    "volatility_regime": "high",
    "trend_strength": 28.4,
    "market_regime": "bull",
    "suggested_position_size": 0.6,
    "stop_loss_atr": 1.5,
    "take_profit_atr": 2.5,
    "risk_reward_ratio": 1.67,
}

MOCK_NEWS: List[Dict[str, Any]] = [
    {
        "date": "2024-06-15",
        "headline": "OPEC+ maintains output cuts through Q3",
        "category": "OPEC",
        "sentiment": "bullish",
    },
    {
        "date": "2024-06-14",
        "headline": "US dollar weakens on Fed pause signals",
        "category": "macro",
        "sentiment": "bullish",
    },
    {
        "date": "2024-06-13",
        "headline": "China demand outlook revised downward",
        "category": "demand",
        "sentiment": "bearish",
    },
]

MOCK_DECISIONS: List[Dict[str, Any]] = [
    {
        "timestamp": "2024-06-12T09:30:00+00:00",
        "decision": "BUY",
        "direction": "UP",
        "confidence": 0.61,
        "position_size": 0.4,
        "stop_loss": 82.5,
        "take_profit": 90.0,
        "rationale": "Trend-following entry on strong model agreement.",
    },
    {
        "timestamp": "2024-06-10T09:30:00+00:00",
        "decision": "HOLD",
        "direction": "NEUTRAL",
        "confidence": 0.48,
        "position_size": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "rationale": "Volatility elevated; waiting for confirmation.",
    },
]

MOCK_SIGNAL: Dict[str, Any] = {
    "confidence_score": 0.68,
    "signal_direction": "UP",
    "model_name": "XGBoost_cls",
    "as_of_date": "2024-06-15",
}

DECISION_SYSTEM_PROMPT = (
    "You are a decision-support agent for an oil trading desk.\n"
    "You receive a machine-learning signal payload.\n"
    "Summarize the signal in 2-3 concise bullet points.\n"
    "End with: SIGNAL_BIAS: UP | DOWN | NEUTRAL (confidence <0-1>)"
)


async def mock_api_diagnostic() -> Dict[str, Any]:
    return MOCK_DIAGNOSTIC


async def mock_api_news() -> List[Dict[str, Any]]:
    return MOCK_NEWS


async def mock_api_decisions() -> List[Dict[str, Any]]:
    return MOCK_DECISIONS


async def mock_api_signal() -> Dict[str, Any]:
    return MOCK_SIGNAL


def _format_signal(signal: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"As of {signal.get('as_of_date', 'n/a')}.",
            f"Model: {signal.get('model_name', 'n/a')}.",
            f"Direction: {signal.get('signal_direction', 'n/a')}.",
            f"Confidence: {signal.get('confidence_score', 'n/a')}.",
        ]
    )


async def decision_agent(signal: Dict[str, Any], cfg: PipelineConfig) -> str:
    llm = get_llm(cfg)
    input_text = "ML SIGNAL:\n" + _format_signal(signal)
    response = await llm.ainvoke(
        [SystemMessage(content=DECISION_SYSTEM_PROMPT), HumanMessage(content=input_text)]
    )
    return response.content if hasattr(response, "content") else str(response)


def _merge_manager_inputs(risk_report: str, decision_report: str) -> str:
    parts: List[str] = []
    if risk_report:
        parts.append("RISK MANAGER REPORT:\n" + risk_report)
    if decision_report:
        parts.append("DECISION AGENT REPORT:\n" + decision_report)
    return "\n\n".join(parts).strip()


async def run_pipeline() -> Dict[str, Any]:
    cfg = PipelineConfig()
    cfg.llm_provider = "ollama"
    cfg.llm_model = "llama3.2"
    cfg.llm_base_url = "http://localhost:11434/v1"
    cfg.llm_api_key = "local"

    diagnostic, news, decisions, signal = await asyncio.gather(
        mock_api_diagnostic(),
        mock_api_news(),
        mock_api_decisions(),
        mock_api_signal(),
    )

    state: Dict[str, Any] = {
        "diagnostic": diagnostic,
        "news": news,
        "decisions_cache": decisions,
        "bullish_report": "",
        "bearish_report": "",
        "trader_report": "",
        "risk_report": "",
        "final_decision": {},
        "error": None,
    }

    bullish_out, bearish_out, risk_out = await asyncio.gather(
        bullish_agent(state, cfg),
        bearish_agent(state, cfg),
        risk_agent(state, cfg),
    )
    state.update(bullish_out)
    state.update(bearish_out)
    state.update(risk_out)

    trader_out = await trader_agent(state, cfg)
    state.update(trader_out)

    decision_report = await decision_agent(signal, cfg)
    merged_risk_report = _merge_manager_inputs(state.get("risk_report", ""), decision_report)

    manager_state = dict(state)
    manager_state["risk_report"] = merged_risk_report
    manager_out = await manager_agent(manager_state, cfg)
    state.update(manager_out)

    print("=== BULLISH ANALYST ===")
    print(state.get("bullish_report", ""))
    print("\n=== BEARISH ANALYST ===")
    print(state.get("bearish_report", ""))
    print("\n=== TRADER ===")
    print(state.get("trader_report", ""))
    print("\n=== RISK MANAGER ===")
    print(state.get("risk_report", ""))
    print("\n=== DECISION AGENT ===")
    print(decision_report)
    print("\n=== MANAGER DECISION ===")
    print(json.dumps(state.get("final_decision", {}), indent=2, ensure_ascii=False))

    return state.get("final_decision", {})


def main() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
