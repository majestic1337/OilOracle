"""Paper trading simulator for agent decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import sqlalchemy
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.pipeline.config import PipelineConfig

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

ADD_ENTRY_PRICE_SQL = """
ALTER TABLE agent_decisions
ADD COLUMN IF NOT EXISTS entry_price FLOAT;
"""


class PaperTradingSimulator:
    """Records trade decisions and tracks PnL in paper trading environment."""

    def __init__(self, cfg: PipelineConfig, engine: sqlalchemy.Engine) -> None:
        self.cfg = cfg
        self.engine = engine
        with self.engine.begin() as conn:
            conn.execute(text(CREATE_TABLE_SQL))
            conn.execute(text(ADD_ENTRY_PRICE_SQL))

    def execute(
        self,
        decision: Dict[str, Any],
        current_price: float,
    ) -> Dict[str, Any]:
        """Save decision to DB. Returns execution record with entry details."""
        timestamp = decision.get("timestamp")
        if timestamp:
            try:
                parsed_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                parsed_ts = datetime.now(timezone.utc)
        else:
            parsed_ts = datetime.now(timezone.utc)

        record = {
            "timestamp": parsed_ts,
            "decision": decision.get("decision"),
            "direction": decision.get("direction"),
            "confidence": decision.get("confidence"),
            "position_size": decision.get("position_size"),
            "stop_loss": decision.get("stop_loss"),
            "take_profit": decision.get("take_profit"),
            "rationale": decision.get("rationale"),
            "entry_price": current_price if decision.get("decision") in {"BUY", "SELL"} else None,
        }

        insert_sql = text(
            """
            INSERT INTO agent_decisions (
                timestamp, decision, direction, confidence, position_size,
                stop_loss, take_profit, rationale, entry_price
            )
            VALUES (
                :timestamp, :decision, :direction, :confidence, :position_size,
                :stop_loss, :take_profit, :rationale, :entry_price
            )
            RETURNING id
            """
        )

        with self.engine.begin() as conn:
            result = conn.execute(insert_sql, record)
            decision_id = result.scalar_one()

        return {
            "id": decision_id,
            "timestamp": parsed_ts.isoformat(),
            "decision": record["decision"],
            "direction": record["direction"],
            "entry_price": record["entry_price"],
            "position_size": record["position_size"],
        }

    def update_pnl(
        self,
        decision_id: int,
        current_price: float,
    ) -> float:
        """Check stop_loss / take_profit. Update pnl in DB. Return current pnl."""
        fetch_sql = text(
            """
            SELECT decision, position_size, stop_loss, take_profit, entry_price, pnl, closed_at
            FROM agent_decisions
            WHERE id = :id
            """
        )
        with self.engine.begin() as conn:
            row = conn.execute(fetch_sql, {"id": decision_id}).mappings().first()
            if not row:
                return 0.0

            decision = row.get("decision")
            entry_price = row.get("entry_price")
            position_size = row.get("position_size") or 0.0
            stop_loss = row.get("stop_loss")
            take_profit = row.get("take_profit")

            if not entry_price or decision not in {"BUY", "SELL"}:
                return float(row.get("pnl") or 0.0)

            notional = self.cfg.initial_capital * position_size
            if entry_price == 0:
                pnl = 0.0
            elif decision == "BUY":
                pnl = ((current_price - entry_price) / entry_price) * notional
            else:
                pnl = ((entry_price - current_price) / entry_price) * notional

            should_close = False
            if decision == "BUY":
                if stop_loss is not None and current_price <= stop_loss:
                    should_close = True
                if take_profit is not None and current_price >= take_profit:
                    should_close = True
            elif decision == "SELL":
                if stop_loss is not None and current_price >= stop_loss:
                    should_close = True
                if take_profit is not None and current_price <= take_profit:
                    should_close = True

            update_sql = text(
                """
                UPDATE agent_decisions
                SET pnl = :pnl,
                    closed_at = CASE WHEN :closed THEN :closed_at ELSE closed_at END
                WHERE id = :id
                """
            )
            conn.execute(
                update_sql,
                {
                    "pnl": pnl,
                    "closed": should_close,
                    "closed_at": datetime.now(timezone.utc) if should_close else None,
                    "id": decision_id,
                },
            )

            return float(pnl)

    def get_summary(self) -> Dict[str, Any]:
        """Return: total_pnl, win_rate, total_trades, open_positions."""
        try:
            with self.engine.begin() as conn:
                total_trades = conn.execute(
                    text("SELECT COUNT(*) FROM agent_decisions")
                ).scalar_one()
                total_pnl = conn.execute(
                    text("SELECT COALESCE(SUM(pnl), 0) FROM agent_decisions")
                ).scalar_one()
                wins = conn.execute(
                    text("SELECT COUNT(*) FROM agent_decisions WHERE pnl > 0")
                ).scalar_one()
                open_positions = conn.execute(
                    text("SELECT COUNT(*) FROM agent_decisions WHERE closed_at IS NULL")
                ).scalar_one()

            win_rate = float(wins) / total_trades if total_trades else 0.0
            return {
                "total_pnl": float(total_pnl),
                "win_rate": win_rate,
                "total_trades": int(total_trades),
                "open_positions": int(open_positions),
            }
        except SQLAlchemyError:
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "open_positions": 0,
            }
