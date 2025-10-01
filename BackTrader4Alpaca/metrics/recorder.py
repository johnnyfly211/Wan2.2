"""Utility for collecting run metrics and exposing convenient lookups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class MetricsRecorder:
    """Capture realised and unrealised P&L figures for reporting."""

    realized_pnl: float = 0.0
    unrealized_pnl_eor: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def track_realized_pnl(self, value: float) -> None:
        self.realized_pnl += value

    def track_unrealized_end_of_run(self, value: float) -> None:
        self.unrealized_pnl_eor += value
        self.extra_metrics["unrealized_pnl_eor"] = self.unrealized_pnl_eor

    def snapshot(self) -> Dict[str, float]:
        payload = {
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl_eor": round(self.unrealized_pnl_eor, 2),
        }
        payload.update({key: round(val, 2) for key, val in self.extra_metrics.items()})
        return payload


__all__ = ["MetricsRecorder"]
