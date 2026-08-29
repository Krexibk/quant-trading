"""Performance statistics."""

import numpy as np
import pandas as pd
import pytest

from quantlab.metrics import (
    drawdown_series,
    max_drawdown_duration,
    monthly_returns,
    performance_stats,
)


@pytest.fixture
def steady():
    idx = pd.bdate_range("2020-01-01", periods=504)
    return pd.Series(10_000 * (1.0004 ** np.arange(504)), index=idx)


def test_total_return_and_cagr(steady):
    s = performance_stats(steady)
    assert s.total_return == pytest.approx(steady.iloc[-1] / 10_000 - 1)
    assert s.cagr == pytest.approx(0.1058, abs=0.01)


def test_zero_volatility_gives_no_sharpe(steady):
    """A perfectly smooth curve has zero downside deviation.

    The naive formula divides by zero here; the result must be a finite
    number, not inf or NaN, or the API cannot serialise it.
    """
    s = performance_stats(steady)
    assert np.isfinite(s.sharpe) and np.isfinite(s.sortino)


def test_drawdown_is_non_positive():
    idx = pd.bdate_range("2020-01-01", periods=100)
    eq = pd.Series(np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 120, 50)]), index=idx)
    dd = drawdown_series(eq)
    assert (dd <= 1e-12).all()
    assert dd.min() == pytest.approx(-0.2, abs=0.01)


def test_drawdown_duration():
    eq = pd.Series([100, 90, 80, 95, 105, 110])
    assert max_drawdown_duration(eq) == 3


def test_flat_equity_has_no_drawdown():
    assert max_drawdown_duration(pd.Series([100.0] * 10)) == 0


def test_trade_based_stats_override_bar_stats():
    idx = pd.bdate_range("2020-01-01", periods=50)
    eq = pd.Series(np.linspace(10_000, 11_000, 50), index=idx)
    s = performance_stats(eq, trade_pnls=[100.0, -50.0, 200.0, -25.0])
    assert s.trades == 4
    assert s.hit_rate == pytest.approx(0.5)
    assert s.profit_factor == pytest.approx(300 / 75)


def test_all_winning_trades_profit_factor():
    idx = pd.bdate_range("2020-01-01", periods=10)
    eq = pd.Series(np.linspace(100, 110, 10), index=idx)
    assert performance_stats(eq, trade_pnls=[10.0, 20.0]).profit_factor == 0.0


def test_empty_and_single_point_series():
    assert performance_stats(pd.Series(dtype=float)).days == 0
    assert performance_stats(pd.Series([100.0])).days == 1


def test_stats_are_json_safe():
    import json
    idx = pd.bdate_range("2020-01-01", periods=30)
    eq = pd.Series(np.linspace(100, 100, 30), index=idx)
    json.dumps(performance_stats(eq).to_dict())


def test_monthly_returns_table(steady):
    t = monthly_returns(steady)
    assert not t.empty and t.shape[1] <= 12
