from pathlib import Path

from BackTrader4Alpaca.cli.init_wizard import Blueprint, BlueprintStore, InitWizard


class DummyInput:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return ""


def test_build_workflow_persists_auto_liquidation_toggle(tmp_path: Path):
    inputs = DummyInput([
        "build",  # mode
        "AAPL,MSFT",  # symbols
        "BackTrader4Alpaca.strategies.sma_crossover.SmaCrossStrategy",  # strategy
        "1d",  # timeframe
        "10000",  # portfolio
        "n",  # auto-liquidation toggle
        "{}",  # fees
        "",  # auto-name
    ])
    store = BlueprintStore(root=tmp_path)
    wizard = InitWizard(store=store, input_fn=inputs)
    result = wizard.run()

    assert result.runtime_config["auto_liquidate_on_close"] is False
    saved_files = list(tmp_path.glob("*.yml"))
    assert saved_files


def test_load_workflow_can_skip_edits(tmp_path: Path):
    store = BlueprintStore(root=tmp_path)
    store.save(
        Blueprint(
            name="example",
            payload={
                "symbols": "AAPL",
                "strategy": "BackTrader4Alpaca.strategies.sma_crossover.SmaCrossStrategy",
                "timeframe": "1d",
                "portfolio": "10000",
                "auto_liquidate_on_close": True,
            },
        )
    )

    inputs = DummyInput([
        "load",  # mode
        "1",  # select blueprint
        "n",  # skip edits
    ])
    wizard = InitWizard(store=store, input_fn=inputs)
    result = wizard.run()
    assert result.runtime_config["auto_liquidate_on_close"] is True


def test_load_workflow_allows_edits(tmp_path: Path):
    store = BlueprintStore(root=tmp_path)
    store.save(
        Blueprint(
            name="example",
            payload={
                "symbols": "AAPL",
                "strategy": "BackTrader4Alpaca.strategies.sma_crossover.SmaCrossStrategy",
                "timeframe": "1d",
                "portfolio": "10000",
                "auto_liquidate_on_close": False,
            },
        )
    )

    inputs = DummyInput([
        "load",  # mode
        "1",  # select blueprint
        "y",  # edit
        "AAPL,MSFT",  # symbols
        "BackTrader4Alpaca.strategies.composed.ComposedStrategy",  # strategy
        "4h",  # timeframe
        "25000",  # portfolio
        "y",  # auto liquidation
        "{}",  # fees
    ])
    wizard = InitWizard(store=store, input_fn=inputs)
    result = wizard.run()
    assert result.runtime_config["symbols"] == "AAPL,MSFT"
    assert result.runtime_config["auto_liquidate_on_close"] is True
