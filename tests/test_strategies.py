"""Every registered strategy must behave itself on arbitrary data."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import run_backtest
from quantlab.strategies import build_pair_frame, get_strategy, list_strategies

SINGLE = [s for s in list_strategies() if not s.needs_pair]
IDS = [s.name for s in SINGLE]


@pytest.mark.parametrize("strat", SINGLE, ids=IDS)
def test_signal_is_well_formed(strat, prices):
    sig = strat.generate(prices)["signal"]
    assert len(sig) == len(prices)
    assert sig.index.equals(prices.index)
    assert not sig.isna().any()
    assert sig.between(-1.0, 1.0).all()


@pytest.mark.parametrize("strat", SINGLE, ids=IDS)
def test_strategy_backtests(strat, prices):
    r = run_backtest(strat.name, prices, symbol="TEST")
    assert len(r.equity) == len(prices)
    assert (r.equity > 0).all()
    assert np.isfinite(r.stats.sharpe)


@pytest.mark.parametrize("strat", SINGLE, ids=IDS)
def test_allow_short_zero_never_goes_short(strat, prices):
    if "allow_short" not in strat.defaults:
        pytest.skip("no allow_short parameter")
    sig = strat.generate(prices, allow_short=0)["signal"]
    assert (sig >= 0).all()


@pytest.mark.parametrize("strat", SINGLE, ids=IDS)
def test_survives_flat_prices(strat):
    """A constant series has zero volatility and zero range.

    Every indicator that divides by a standard deviation, an ATR or a band
    width can produce inf/NaN here. The strategy must return a clean signal
    instead of propagating them.
    """
    idx = pd.bdate_range("2021-01-01", periods=300)
    flat = pd.DataFrame({"Open": 50.0, "High": 50.0, "Low": 50.0,
                         "Close": 50.0, "Volume": 1e6}, index=idx)
    sig = strat.generate(flat)["signal"]
    assert not sig.isna().any()
    assert np.isfinite(sig).all()


@pytest.mark.parametrize("strat", SINGLE, ids=IDS)
def test_survives_short_history(strat):
    idx = pd.bdate_range("2021-01-01", periods=40)
    rng = np.random.default_rng(5)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 40)), index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": 1e6}, index=idx)
    sig = strat.generate(df)["signal"]
    assert not sig.isna().any()


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        get_strategy("does-not-exist")


def test_lookup_is_normalised():
    assert get_strategy("MACD").name == "macd"
    assert get_strategy("heikin-ashi").name == "heikin_ashi"
    assert get_strategy(" Dual Thrust ").name == "dual_thrust"


def test_macd_swaps_inverted_spans(prices):
    """fast > slow is nonsense; the strategy should normalise rather than
    silently produce an inverted signal."""
    a = get_strategy("macd").generate(prices, fast=26, slow=12)["signal"]
    b = get_strategy("macd").generate(prices, fast=12, slow=26)["signal"]
    assert a.equals(b)


def test_pairs_needs_second_leg(prices):
    with pytest.raises(ValueError, match="Close_b"):
        get_strategy("pairs").generate(prices)


def test_pairs_runs(prices, prices_b):
    frame = build_pair_frame(prices, prices_b)
    r = run_backtest("pairs", frame, symbol="A/B")
    assert (r.equity > 0).all()
    assert r.signals["zscore"].notna().any()


def test_pairs_hedge_ratio_is_rolling(prices, prices_b):
    """The hedge ratio must not be fitted on the whole sample.

    An in-sample beta is lookahead: recomputing on a truncated history has to
    give the same values for the overlapping dates.
    """
    frame = build_pair_frame(prices, prices_b)
    full = get_strategy("pairs").generate(frame)["beta"]
    part = get_strategy("pairs").generate(frame.iloc[:600])["beta"]
    common = full.iloc[:600].dropna().index.intersection(part.dropna().index)
    assert len(common) > 100
    assert np.allclose(full.loc[common], part.loc[common])


def test_strategies_expose_ui_metadata():
    for s in list_strategies():
        d = s.to_dict()
        assert d["name"] and d["label"] and d["description"]
        for p in d["params"]:
            assert p["name"] and p["default"] is not None
