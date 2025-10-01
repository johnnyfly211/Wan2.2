"""Helpers to export equity curves while ignoring synthetic open rows."""

from __future__ import annotations

from typing import Iterable, List

from BackTrader4Alpaca.analyzers.trade_logger import TradeRecord


def realised_equity_curve(records: Iterable[TradeRecord]) -> List[float]:
    curve: List[float] = []
    equity = 0.0
    for record in records:
        if record.direction == "Open End of Run":
            continue
        equity += record.realized_pnl
        curve.append(equity)
    return curve


__all__ = ["realised_equity_curve"]
