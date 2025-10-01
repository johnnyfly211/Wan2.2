"""Composite strategy capable of orchestrating multiple entry/exit blocks."""

from __future__ import annotations

from typing import Iterable, List

from BackTrader4Alpaca.analyzers.trade_logger import PositionSnapshot
from BackTrader4Alpaca.strategies.base import StrategyBase


class ComposedStrategy(StrategyBase):
    """Wrapper coordinating multiple child strategies."""

    def __init__(self, *strategies: StrategyBase, **kwargs) -> None:
        super().__init__(**kwargs)
        self.children: List[StrategyBase] = list(strategies)

    def on_child_signal(self, child: StrategyBase, snapshot: PositionSnapshot) -> None:
        self.open_position(snapshot)

    def finalise_run(self, *, as_of: str | None = None) -> None:  # type: ignore[override]
        for child in self.children:
            child.finalise_run(as_of=as_of)
        super().finalise_run(as_of=as_of)

    def iter_children(self) -> Iterable[StrategyBase]:
        return tuple(self.children)


__all__ = ["ComposedStrategy"]
