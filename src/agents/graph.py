"""LangGraph orchestration for the multi-agent oil analysis team."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from src.agents.bearish import bearish_agent
from src.agents.bullish import bullish_agent
from src.agents.manager import manager_agent
from src.agents.risk import risk_agent
from src.agents.trader import trader_agent
from src.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    diagnostic: Dict[str, Any]
    news: List[Dict[str, Any]]
    decisions_cache: List[Dict[str, Any]]
    bullish_report: str
    bearish_report: str
    trader_report: str
    risk_report: str
    final_decision: Dict[str, Any]
    error: str | None


MOCK_DIAGNOSTIC = {
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

MOCK_NEWS = [
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

LAST_AGENT_STATE: AgentState | None = None


def get_last_state() -> AgentState | None:
    """Return the most recent agent state from run_agent_team."""
    return LAST_AGENT_STATE


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    fallback: Dict[str, Any],
    label: str,
) -> Dict[str, Any]:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as exc:
        logger.warning("%s fetch failed (%s). Using mock data.", label, exc)
        return fallback
    except httpx.HTTPStatusError as exc:
        logger.warning("%s returned %s. Using mock data.", label, exc.response.status_code)
        return fallback
    except ValueError as exc:
        logger.warning("%s returned invalid JSON (%s). Using mock data.", label, exc)
        return fallback


async def _build_initial_state(cfg: PipelineConfig) -> AgentState:
    base_url = cfg.api_base_url.rstrip("/")

    async with httpx.AsyncClient(timeout=10.0) as client:
        diagnostic = await _fetch_json(
            client,
            f"{base_url}/market/diagnostic",
            MOCK_DIAGNOSTIC,
            "diagnostic",
        )
        news_payload = await _fetch_json(
            client,
            f"{base_url}/news/recent",
            {"news": MOCK_NEWS},
            "news",
        )
        decisions_payload = await _fetch_json(
            client,
            f"{base_url}/decisions/history?days={cfg.decisions_cache_days}",
            {"decisions": []},
            "decisions",
        )

    return {
        "diagnostic": diagnostic,
        "news": news_payload.get("news", []),
        "decisions_cache": decisions_payload.get("decisions", []),
        "bullish_report": "",
        "bearish_report": "",
        "trader_report": "",
        "risk_report": "",
        "final_decision": {},
        "error": None,
    }


def _build_graph(cfg: PipelineConfig):
    graph = StateGraph(AgentState)

    async def bullish_node(state: AgentState) -> AgentState:
        return await bullish_agent(state, cfg)

    async def bearish_node(state: AgentState) -> AgentState:
        return await bearish_agent(state, cfg)

    async def trader_node(state: AgentState) -> AgentState:
        return await trader_agent(state, cfg)

    async def risk_node(state: AgentState) -> AgentState:
        return await risk_agent(state, cfg)

    async def manager_node(state: AgentState) -> AgentState:
        return await manager_agent(state, cfg)

    async def execution_module(state: AgentState) -> AgentState:
        return state

    graph.add_node("bullish", bullish_node)
    graph.add_node("bearish", bearish_node)
    graph.add_node("trader", trader_node)
    graph.add_node("risk", risk_node)
    graph.add_node("manager", manager_node)
    graph.add_node("execution", execution_module)

    graph.add_edge(START, "bullish")
    graph.add_edge(START, "bearish")
    graph.add_edge("bullish", "trader")
    graph.add_edge("bearish", "trader")
    graph.add_edge("bullish", "risk")
    graph.add_edge("bearish", "risk")
    graph.add_edge("trader", "manager")
    graph.add_edge("risk", "manager")
    graph.add_edge("manager", "execution")
    graph.add_edge("execution", END)

    return graph.compile()


async def _post_final_decision(cfg: PipelineConfig, decision: Dict[str, Any]) -> None:
    base_url = cfg.api_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(f"{base_url}/decisions/save", json=decision)
        except httpx.RequestError as exc:
            logger.warning("Unable to post decision (%s).", exc)


async def run_agent_team(cfg: PipelineConfig) -> Dict[str, Any]:
    """Fetch data, run the LangGraph agent team, save and return final decision."""
    global LAST_AGENT_STATE

    initial_state = await _build_initial_state(cfg)
    graph = _build_graph(cfg)
    final_state = await graph.ainvoke(initial_state)

    LAST_AGENT_STATE = final_state
    final_decision = final_state.get("final_decision", {})

    await _post_final_decision(cfg, final_decision)

    return final_decision
