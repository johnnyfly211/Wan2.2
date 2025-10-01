"""Trade logging helpers used across strategies and analyzers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable, List, Optional


@dataclass(slots=True)
class PositionSnapshot:
    """Snapshot of an open position used for logging and metrics.

    Parameters
    ----------
    trade_id:
        Identifier of the trade that opened the position.  This is reused when
        we create an ``Open End of Run`` entry so downstream tooling can easily
        collapse the lifecycle of a trade.
    symbol:
        Trading symbol for the position.
    direction:
        ``"LONG"`` or ``"SHORT"``.
    quantity:
        Number of shares/contracts currently held.
    entry_price:
        Price of the original fill.
    last_price:
        Latest observed mark used when calculating unrealised P&L.
    """

    trade_id: str
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    last_price: float

    def close(self, price: float) -> "PositionSnapshot":
        """Return a copy representing a closed trade at ``price``."""

        return PositionSnapshot(
            trade_id=self.trade_id,
            symbol=self.symbol,
            direction=self.direction,
            quantity=self.quantity,
            entry_price=self.entry_price,
            last_price=price,
        )

    @property
    def value(self) -> float:
        return self.quantity * self.last_price

    @property
    def realized_pnl(self) -> float:
        multiplier = 1 if self.direction.upper() == "LONG" else -1
        return (self.last_price - self.entry_price) * self.quantity * multiplier

    @property
    def unrealized_pnl(self) -> float:
        return self.realized_pnl


@dataclass(slots=True)
class TradeRecord:
    trade_id: str
    symbol: str
    direction: str
    quantity: float
    price: float
    value: float
    realized_pnl: float
    unrealized_pnl: float
    timestamp: datetime
    status: str = "CLOSED"


class TradeLogger:
    """Collects trade lifecycle events for persistence and analysis."""

    def __init__(self) -> None:
        self.records: List[TradeRecord] = []

    # ------------------------------------------------------------------
    # Core trade logging
    # ------------------------------------------------------------------
    def _append(self, record: TradeRecord) -> None:
        self.records.append(record)

    def log_open(self, snapshot: PositionSnapshot) -> None:
        self._append(
            TradeRecord(
                trade_id=snapshot.trade_id,
                symbol=snapshot.symbol,
                direction=snapshot.direction,
                quantity=snapshot.quantity,
                price=snapshot.entry_price,
                value=snapshot.entry_price * snapshot.quantity,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                timestamp=datetime.now(UTC),
                status="OPEN",
            )
        )

    def log_close(self, snapshot: PositionSnapshot) -> None:
        self._append(
            TradeRecord(
                trade_id=snapshot.trade_id,
                symbol=snapshot.symbol,
                direction="CLOSE",
                quantity=snapshot.quantity,
                price=snapshot.last_price,
                value=snapshot.value,
                realized_pnl=snapshot.realized_pnl,
                unrealized_pnl=0.0,
                timestamp=datetime.now(UTC),
                status="CLOSED",
            )
        )

    # ------------------------------------------------------------------
    # End-of-run handling
    # ------------------------------------------------------------------
    def log_open_positions_end_of_run(
        self,
        positions: Iterable[PositionSnapshot],
        *,
        as_of: Optional[str] = None,
    ) -> float:
        """Record synthetic rows for open positions at the end of a run.

        Returns the aggregated unrealised P&L so callers can pipe the value into
        new metrics such as ``unrealized_pnl_eor``.
        """

        total_unrealized = 0.0
        for snapshot in positions:
            unrealized = snapshot.unrealized_pnl
            total_unrealized += unrealized
            self._append(
                TradeRecord(
                    trade_id=snapshot.trade_id,
                    symbol=snapshot.symbol,
                    direction="Open End of Run",
                    quantity=snapshot.quantity,
                    price=snapshot.last_price,
                    value=snapshot.value,
                    realized_pnl=0.0,
                    unrealized_pnl=unrealized,
                    timestamp=datetime.now(UTC),
                    status="OPEN_EOR" if not as_of else f"OPEN_EOR@{as_of}",
                )
            )
        return total_unrealized

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def open_end_of_run_records(self) -> List[TradeRecord]:
        return [record for record in self.records if record.direction == "Open End of Run"]


__all__ = ["PositionSnapshot", "TradeLogger", "TradeRecord"]
