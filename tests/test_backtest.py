"""Engine correctness: accounting, costs, risk controls and lookahead."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import Backtester, RiskConfig, run_backtest
from quantlab.config import CostModel

ZERO = CostModel(0.0, 0.0, 0.0)
FLAT = RiskConfig(target_volatility=None, max_leverage=1.0, stop_loss_atr=None,
                  trailing_atr=None, rebalance_threshold=0.0)


def _long(prices):
    return pd.Series(1.0, index=prices.index)


def test_always_long_tracks_buy_and_hold(prices):
    r = Backtester(100_000, ZERO, FLAT).run(prices, _long(prices))
    assert r.stats.total_return == pytest.approx(r.benchmark_stats.total_return, abs=0.05)


def test_flat_signal_preserves_capital(prices):
    r = Backtester(100_000, ZERO, FLAT).run(prices, pd.Series(0.0, index=prices.index))
    assert r.stats.total_return == pytest.approx(0.0, abs=1e-9)
    assert r.equity.iloc[-1] == pytest.approx(100_000)


def test_same_bar_signal_is_not_profitable(prices):
    """The classic lookahead bug must not pay.

    Trading the sign of *today's* return can only work if the engine lets a
    signal act on the bar it was computed from. It must not.
    """
    same_bar = np.sign(prices["Close"].pct_change()).fillna(0.0)
    r = Backtester(100_000, ZERO, FLAT).run(prices, same_bar)
    assert r.stats.total_return < 1.0


def test_perfect_foresight_is_profitable(prices):
    """Sanity check the mirror image: a genuinely prescient signal must win,
    otherwise the previous test would pass for the wrong reason."""
    tomorrow = np.sign(prices["Close"].pct_change().shift(-1)).fillna(0.0)
    r = Backtester(100_000, ZERO, FLAT).run(prices, tomorrow)
    assert r.stats.total_return > 1.0


def test_costs_reduce_returns(prices):
    churn = pd.Series(
        np.random.default_rng(7).choice([-1.0, 1.0], len(prices)), index=prices.index
    )
    free = Backtester(100_000, ZERO, FLAT).run(prices, churn).stats.total_return
    paid = Backtester(100_000, CostModel(5, 5, 0), FLAT).run(prices, churn).stats.total_return
    assert paid < free


def test_stop_loss_reduces_drawdown(prices):
    no_stop = Backtester(100_000, ZERO, RiskConfig(None, 20, 1.0, None, None, None))
    stopped = Backtester(100_000, ZERO, RiskConfig(None, 20, 1.0, 1.5, None, None))
    assert stopped.run(prices, _long(prices)).stats.max_drawdown > \
        no_stop.run(prices, _long(prices)).stats.max_drawdown


def test_stop_does_not_immediately_reenter(prices):
    """After a stop-out the engine must wait for the signal to reset.

    Without the lockout a constant signal re-buys at the next open, which
    makes the stop a pure cost with no risk benefit.
    """
    r = Backtester(100_000, ZERO, RiskConfig(None, 20, 1.0, 1.5, None, None)).run(
        prices, _long(prices)
    )
    stops = [t for t in r.trades if t.exit_reason == "stop"]
    assert stops, "expected at least one stop-out"
    assert len(r.trades) < 20


def test_vol_targeting_moves_toward_target(prices):
    r = Backtester(100_000, ZERO, RiskConfig(target_volatility=0.10, max_leverage=3.0,
                                             stop_loss_atr=None)).run(prices, _long(prices))
    assert 0.03 < r.stats.volatility < 0.22


def test_leverage_is_capped(prices):
    r = Backtester(100_000, ZERO, RiskConfig(target_volatility=5.0, max_leverage=1.5,
                                             stop_loss_atr=None)).run(prices, _long(prices))
    assert r.position.abs().max() <= 1.75  # cap plus intraday drift


def test_equity_never_negative(prices):
    churn = pd.Series(
        np.random.default_rng(3).choice([-1.0, 1.0], len(prices)), index=prices.index
    )
    r = Backtester(10_000, CostModel(50, 50, 0), RiskConfig(None, 20, 1.0)).run(prices, churn)
    assert (r.equity >= 0).all()


def test_trades_reconcile_with_stats(prices):
    r = run_backtest("macd", prices, symbol="TEST")
    assert r.stats.trades == len(r.trades)
    if r.trades:
        wins = sum(1 for t in r.trades if t.pnl > 0)
        assert r.stats.hit_rate == pytest.approx(wins / len(r.trades))


def test_short_position_profits_when_price_falls():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(np.linspace(100, 60, 60), index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.005,
                       "Low": close * 0.995, "Close": close})
    r = Backtester(100_000, ZERO, FLAT).run(df, pd.Series(-1.0, index=idx))
    assert r.stats.total_return > 0.1


def test_rejects_too_short_history():
    df = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0]},
                      index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError):
        Backtester().run(df, pd.Series([0.0], index=df.index))


def test_result_serialises(prices):
    d = run_backtest("bollinger", prices, symbol="TEST").to_dict()
    assert {"stats", "equity", "dates", "trades", "drawdown"} <= set(d)
    assert len(d["equity"]) == len(d["dates"]) == len(d["drawdown"])
    import json
    json.dumps(d)  # must contain no NaN/inf that would break the API
