"""Shared helpers for BackTrader strategies.

The real project wires strategies into Backtrader's ``Cerebro`` runtime.  Our
light-weight abstraction keeps the public surface area similar so unit tests and
scripts can exercise end-of-run behaviour without spinning up a full Backtrader
engine.  The primary purpose of this module is to centralise the logic that
handles open positions at the end of a run now that automatic liquidation is
optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from BackTrader4Alpaca.analyzers.trade_logger import PositionSnapshot, TradeLogger
from BackTrader4Alpaca.cerebro_factory import RuntimeConfig
from BackTrader4Alpaca.metrics.recorder import MetricsRecorder


@dataclass(slots=True)
class StrategyState:
    """Simple container tracking open positions by symbol."""

    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)

    def iter_open_positions(self) -> Iterable[PositionSnapshot]:
        return self.positions.values()

    def register_position(self, snapshot: PositionSnapshot) -> None:
        self.positions[snapshot.symbol] = snapshot

    def close_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)


class StrategyBase:
    """Base class shared between all strategies.

    Strategies only need to provide hooks for order generation.  The base class
    manages open position bookkeeping and delegates persistence/logging to the
    :class:`TradeLogger` so end-of-run handling remains consistent across
    strategies.
    """

    def __init__(
        self,
        runtime: RuntimeConfig,
        trade_logger: Optional[TradeLogger] = None,
        metrics: Optional[MetricsRecorder] = None,
    ) -> None:
        self.runtime = runtime
        self.trade_logger = trade_logger or TradeLogger()
        self.metrics = metrics or MetricsRecorder()
        self.state = StrategyState()

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------
    def open_position(self, snapshot: PositionSnapshot) -> None:
        """Record a newly opened position.

        Real strategies will call this helper once their first fill is confirmed.
        Tests can use it directly to seed strategy state before triggering an
        end-of-run event.
        """

        self.state.register_position(snapshot)
        self.trade_logger.log_open(snapshot)

    def close_position(self, symbol: str, price: float) -> None:
        snapshot = self.state.positions.get(symbol)
        if not snapshot:
            return

        closed = snapshot.close(price)
        self.state.close_position(symbol)
        self.trade_logger.log_close(closed)
        self.metrics.track_realized_pnl(closed.realized_pnl)

    # ------------------------------------------------------------------
    # End-of-run lifecycle
    # ------------------------------------------------------------------
    def finalise_run(self, *, as_of: Optional[str] = None) -> None:
        """Handle the end-of-run logic for the strategy.

        When ``auto_liquidate_on_close`` is disabled we snapshot open positions
        and emit ``Open End of Run`` rows through the trade logger.  Strategies
        that opt back into the legacy behaviour will see their remaining
        positions closed using the last available price.
        """

        open_positions = list(self.state.iter_open_positions())
        if not open_positions:
            return

        if self.runtime.auto_liquidate_on_close:
            for position in open_positions:
                self.close_position(position.symbol, position.last_price)
            return

        unrealized_total = self.trade_logger.log_open_positions_end_of_run(
            open_positions,
            as_of=as_of,
        )
        self.metrics.track_unrealized_end_of_run(unrealized_total)


__all__ = ["StrategyBase", "StrategyState"]
