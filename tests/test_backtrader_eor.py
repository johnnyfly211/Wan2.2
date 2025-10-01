from BackTrader4Alpaca.analyzers.trade_logger import PositionSnapshot, TradeLogger
from BackTrader4Alpaca.cerebro_factory import RuntimeConfig
from BackTrader4Alpaca.loggers.validate_logger import ValidateLogger
from BackTrader4Alpaca.metrics.recorder import MetricsRecorder
from BackTrader4Alpaca.strategies.base import StrategyBase


class DummyStrategy(StrategyBase):
    pass


def test_end_of_run_snapshots_when_auto_liquidation_disabled():
    runtime = RuntimeConfig(auto_liquidate_on_close=False)
    logger = TradeLogger()
    metrics = MetricsRecorder()
    strategy = DummyStrategy(runtime=runtime, trade_logger=logger, metrics=metrics)

    strategy.open_position(
        PositionSnapshot(
            trade_id="T1",
            symbol="AAPL",
            direction="LONG",
            quantity=10,
            entry_price=100.0,
            last_price=110.0,
        )
    )
    strategy.finalise_run(as_of="2024-01-01")

    open_rows = logger.open_end_of_run_records()
    assert len(open_rows) == 1
    assert open_rows[0].direction == "Open End of Run"
    assert open_rows[0].unrealized_pnl == 100.0
    assert metrics.unrealized_pnl_eor == 100.0


def test_end_of_run_liquidates_when_flag_enabled():
    runtime = RuntimeConfig(auto_liquidate_on_close=True)
    logger = TradeLogger()
    metrics = MetricsRecorder()
    strategy = DummyStrategy(runtime=runtime, trade_logger=logger, metrics=metrics)

    strategy.open_position(
        PositionSnapshot(
            trade_id="T2",
            symbol="MSFT",
            direction="LONG",
            quantity=5,
            entry_price=200.0,
            last_price=210.0,
        )
    )
    strategy.finalise_run()

    assert not logger.open_end_of_run_records()
    close_records = [record for record in logger.records if record.direction == "CLOSE"]
    assert len(close_records) == 1
    assert metrics.realized_pnl == 50.0


def test_validate_logger_renders_open_positions():
    logger = ValidateLogger()
    trade_logger = TradeLogger()
    trade_logger.log_open_positions_end_of_run(
        [
            PositionSnapshot(
                trade_id="T3",
                symbol="TSLA",
                direction="LONG",
                quantity=2,
                entry_price=300.0,
                last_price=330.0,
            )
        ]
    )
    logger.log_open_positions(trade_logger.open_end_of_run_records())
    output = logger.dump()
    assert "Open at End of Run" in output
    assert "TSLA" in output
