"""CLI entry points."""

import json

import pytest

from quantlab.cli import main

BASE = ["--source", "synthetic", "--symbol", "AAPL", "--start", "2021-01-01", "--end", "2024-12-31"]


def test_strategies_listing(capsys):
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "macd" in out and "bollinger" in out


def test_backtest_human_output(capsys):
    assert main(["backtest", "macd", *BASE]) == 0
    out = capsys.readouterr().out
    assert "Sharpe" in out and "buy & hold" in out and "Max drawdown" in out


def test_backtest_json_output(capsys):
    assert main(["backtest", "rsi", *BASE, "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert {"sharpe", "total_return", "max_drawdown"} <= set(stats)


def test_backtest_with_params(capsys):
    assert main(["backtest", "macd", *BASE, "--params", '{"fast": 8, "slow": 21}']) == 0
    assert "macd" in capsys.readouterr().out


def test_backtest_with_trades(capsys):
    assert main(["backtest", "heikin_ashi", *BASE, "--trades"]) == 0
    assert "trades" in capsys.readouterr().out.lower()


def test_compare_ranks_strategies(capsys):
    assert main(["compare", *BASE]) == 0
    out = capsys.readouterr().out
    assert "buy & hold" in out and "Sharpe" in out


def test_optimise_reports_out_of_sample(capsys):
    assert main(["optimise", "rsi", *BASE, "--max-combos", "6"]) == 0
    out = capsys.readouterr().out
    assert "OOS Sharpe" in out and "best in-sample" in out


def test_pairs_requires_second_symbol(capsys):
    assert main(["backtest", "pairs", *BASE]) == 1
    assert "symbol-b" in capsys.readouterr().err


def test_pairs_with_second_symbol(capsys):
    assert main(["backtest", "pairs", *BASE, "--symbol-b", "MSFT"]) == 0


def test_unknown_strategy_exits_nonzero(capsys):
    assert main(["backtest", "nope", *BASE]) == 1
    assert "error" in capsys.readouterr().err


def test_account_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QUANTLAB_DB", str(tmp_path / "t.db"))
    import importlib

    from quantlab import config
    importlib.reload(config)
    assert main(["account", "status"]) == 0
    assert "equity" in capsys.readouterr().out


def test_invalid_args_exit():
    with pytest.raises(SystemExit):
        main(["backtest"])
