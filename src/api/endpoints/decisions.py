"""Decision storage endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import sqlalchemy
from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.pipelines.config import PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter()

CFG = PipelineConfig()
ENGINE = sqlalchemy.create_engine(CFG.db_url, future=True)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_decisions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    decision VARCHAR(10),
    direction VARCHAR(10),
    confidence FLOAT,
    position_size FLOAT,
    stop_loss FLOAT,
    take_profit FLOAT,
    rationale TEXT,
    pnl FLOAT DEFAULT NULL,
    closed_at TIMESTAMPTZ DEFAULT NULL
);
"""


@router.get("/decisions/history")
def decisions_history(days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
    """Return last N days of decisions; never raises."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = text(
        """
        SELECT id, timestamp, decision, direction, confidence, position_size,
               stop_loss, take_profit, rationale, pnl, closed_at
        FROM agent_decisions
        WHERE timestamp >= :since
        ORDER BY timestamp DESC
        """
    )
    try:
        with ENGINE.begin() as conn:
            result = conn.execute(query, {"since": since})
            rows = [dict(row) for row in result.mappings().all()]
        return {"decisions": rows}
    except SQLAlchemyError:
        logger.exception("Decision history lookup failed; returning empty list")
        return {"decisions": []}


@router.post("/decisions/save")
def decisions_save(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Save a decision row and return its ID."""
    timestamp = payload.get("timestamp")
    if timestamp:
        try:
            parsed_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            parsed_ts = datetime.now(timezone.utc)
    else:
        parsed_ts = datetime.now(timezone.utc)

    record = {
        "timestamp": parsed_ts,
        "decision": payload.get("decision"),
        "direction": payload.get("direction"),
        "confidence": payload.get("confidence"),
        "position_size": payload.get("position_size"),
        "stop_loss": payload.get("stop_loss"),
        "take_profit": payload.get("take_profit"),
        "rationale": payload.get("rationale"),
    }

    insert_sql = text(
        """
        INSERT INTO agent_decisions (
            timestamp, decision, direction, confidence, position_size,
            stop_loss, take_profit, rationale
        )
        VALUES (
            :timestamp, :decision, :direction, :confidence, :position_size,
            :stop_loss, :take_profit, :rationale
        )
        RETURNING id
        """
    )

    with ENGINE.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        result = conn.execute(insert_sql, record)
        decision_id = result.scalar_one()

    return {"status": "ok", "id": decision_id, "timestamp": parsed_ts.isoformat()}
