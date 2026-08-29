"""Indicators must be correct and must never look into the future."""

import numpy as np
import pandas as pd
import pytest

from quantlab import indicators as ind


def test_sma_matches_manual():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]


def test_ema_first_valid_at_span():
    s = pd.Series(np.arange(50.0))
    e = ind.ema(s, 10)
    assert e.isna().sum() == 9
    assert e.iloc[-1] == pytest.approx(44.5, abs=0.6)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up, 14).dropna().iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(down, 14).dropna().iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_in_range(prices):
    r = ind.rsi(prices["Close"]).dropna()
    assert r.between(0, 100).all()


def test_macd_histogram_is_difference(prices):
    m = ind.macd(prices["Close"])
    assert np.allclose((m["macd"] - m["signal"]).dropna(), m["histogram"].dropna())


def test_bollinger_bands_ordered(prices):
    bb = ind.bollinger(prices["Close"]).dropna()
    assert (bb["upper"] >= bb["mid"]).all()
    assert (bb["mid"] >= bb["lower"]).all()


def test_atr_positive(prices):
    assert (ind.atr(prices).dropna() > 0).all()


def test_true_range_covers_gaps():
    df = pd.DataFrame({"Open": [10, 20], "High": [11, 21], "Low": [9, 19], "Close": [10, 20]})
    # Second bar gaps up from 10 to a 19-21 range: TR must span the gap.
    assert ind.true_range(df).iloc[1] == pytest.approx(11.0)


def test_heikin_ashi_recursion():
    df = pd.DataFrame({
        "Open": [10.0, 11, 12], "High": [11.0, 12, 13],
        "Low": [9.0, 10, 11], "Close": [10.5, 11.5, 12.5],
    })
    ha = ind.heikin_ashi(df)
    assert ha["ha_close"].iloc[0] == pytest.approx((10 + 11 + 9 + 10.5) / 4)
    assert ha["ha_open"].iloc[1] == pytest.approx((ha["ha_open"].iloc[0] + ha["ha_close"].iloc[0]) / 2)


def test_donchian_excludes_current_bar(prices):
    """The channel must be built from prior bars only.

    If today's own high were included, `High > upper` could never be true and
    every breakout test would silently be dead.
    """
    dc = ind.donchian(prices, 20)
    assert (prices["High"] > dc["upper"]).any()


def test_parabolic_sar_flips_and_stays_bounded(prices):
    sar = ind.parabolic_sar(prices)
    assert set(sar["trend"].unique()) <= {1, -1}
    assert (sar["trend"].diff() != 0).sum() > 1
    assert sar["sar"].between(prices["Low"].min() * 0.5, prices["High"].max() * 1.5).all()


def test_adx_range(prices):
    a = ind.adx(prices).dropna()
    assert a["adx"].between(0, 100).all()


def test_zscore_is_standardised(prices):
    z = ind.zscore(prices["Close"], 50).dropna()
    assert abs(z.mean()) < 1.0
    assert z.abs().max() < 20


@pytest.mark.parametrize("fn", [
    lambda d: ind.rsi(d["Close"]),
    lambda d: ind.atr(d),
    lambda d: ind.awesome_oscillator(d),
    lambda d: ind.macd(d["Close"])["histogram"],
    lambda d: ind.adx(d)["adx"],
])
def test_no_lookahead(prices, fn):
    """A value at time t must not change when future bars are appended.

    This is the property that separates a usable indicator from one that
    quietly encodes tomorrow's price.
    """
    cut = 800
    full = fn(prices)
    partial = fn(prices.iloc[:cut])
    a = full.iloc[:cut].dropna()
    b = partial.dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 100
    assert np.allclose(a.loc[common], b.loc[common], equal_nan=True)
