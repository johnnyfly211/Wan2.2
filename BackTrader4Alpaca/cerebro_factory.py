"""Factories and helpers for constructing runtime configuration objects.

This module centralises the hydration of runtime configuration used by
strategies.  Historically the runtime always forced open positions to close at
strategy shutdown which distorted historical metrics.  We now expose a
configuration flag allowing callers to opt-in to the legacy behaviour while
preserving the new default of leaving positions open at the end of a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

_DEFAULT_AUTO_LIQUIDATE = False


@dataclass(slots=True)
class RuntimeConfig:
    """Container for values that influence live strategy behaviour.

    Parameters
    ----------
    auto_liquidate_on_close:
        Controls whether strategies should emit forced closing orders for open
        positions when their execution window ends.  The default is ``False`` to
        match real world behaviour – positions remain open until the user
        explicitly closes them – while retaining a legacy toggle for existing
        automation that still depends on the older workflow.
    extras:
        Additional configuration items that are not interpreted directly by the
        runtime.  The values are preserved so downstream components can access
        them without having to rehydrate configuration themselves.
    """

    auto_liquidate_on_close: bool = _DEFAULT_AUTO_LIQUIDATE
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        hydrated_config: Dict[str, Any],
        *,
        default_auto_liquidate: bool | None = None,
    ) -> "RuntimeConfig":
        """Build a :class:`RuntimeConfig` from a hydrated configuration mapping.

        ``BackTrader4Alpaca`` already performs several rounds of configuration
        merging before strategies are instantiated.  The final dictionary may or
        may not contain the new ``auto_liquidate_on_close`` key so we provide a
        defensive helper that pops the value if present, falls back to a
        sensible default, and returns the remaining items untouched.
        """

        config = dict(hydrated_config or {})
        if default_auto_liquidate is None:
            default_auto_liquidate = _DEFAULT_AUTO_LIQUIDATE

        flag = config.pop("auto_liquidate_on_close", default_auto_liquidate)
        return cls(auto_liquidate_on_close=bool(flag), extras=config)


def build_runtime(config: Dict[str, Any] | None = None) -> RuntimeConfig:
    """Create a :class:`RuntimeConfig` instance from a hydrated configuration.

    The helper mirrors the original ``cerebro_factory`` entry point meaning
    existing callers can import and use the function without code changes.  The
    only behavioural update is the ability to opt-in to legacy auto liquidation
    by passing ``{"auto_liquidate_on_close": True}`` in the config payload.
    """

    return RuntimeConfig.from_dict(config or {})


__all__ = ["RuntimeConfig", "build_runtime"]
