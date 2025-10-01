"""Validation logger enhanced to surface open positions at end-of-run."""

from __future__ import annotations

from typing import Iterable

from BackTrader4Alpaca.analyzers.trade_logger import TradeRecord


class ValidateLogger:
    """Collects human readable lines describing validation outcomes."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def log_open_positions(self, records: Iterable[TradeRecord]) -> None:
        for record in records:
            self.info(
                f"Open at End of Run: {record.symbol} qty={record.quantity} "
                f"price={record.price:.2f} unrealized_pnl={record.unrealized_pnl:.2f}"
            )

    def dump(self) -> str:
        return "\n".join(self.messages)


__all__ = ["ValidateLogger"]
