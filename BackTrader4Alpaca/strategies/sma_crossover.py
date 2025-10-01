"""Example strategy demonstrating end-of-run behaviour changes."""

from __future__ import annotations

from BackTrader4Alpaca.analyzers.trade_logger import PositionSnapshot
from BackTrader4Alpaca.strategies.base import StrategyBase


class SmaCrossStrategy(StrategyBase):
    """Minimal SMA crossover strategy that only tracks positions."""

    def on_signal(self, *, trade_id: str, symbol: str, price: float, size: float) -> None:
        snapshot = PositionSnapshot(
            trade_id=trade_id,
            symbol=symbol,
            direction="LONG" if size > 0 else "SHORT",
            quantity=abs(size),
            entry_price=price,
            last_price=price,
        )
        self.open_position(snapshot)

    def update_price(self, symbol: str, price: float) -> None:
        if symbol in self.state.positions:
            snapshot = self.state.positions[symbol]
            snapshot.last_price = price


__all__ = ["SmaCrossStrategy"]
