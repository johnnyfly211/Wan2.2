# BackTrader4Alpaca Utilities

This lightweight mirror of the internal project exposes the portions of the
runtime required for the kata.  Two high level themes are covered:

1. **End-of-run position handling** – automatic liquidation is now opt-in.  The
   runtime snapshots open positions as `Open End of Run` rows, surfaces an
   `unrealized_pnl_eor` metric, and writes user friendly log lines.
2. **Saved composition workflow** – the wizard now prompts users to build a new
   strategy or load an existing one and offers pre-filled prompts when editing a
   saved composition.

Use ``poetry run python -m BackTrader4Alpaca.cli.init_wizard`` to exercise the
wizard locally. Blueprints are stored as JSON under ``./blueprints`` so existing
files remain compatible while newly created ones capture the additional runtime
toggle.
