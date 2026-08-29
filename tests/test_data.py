"""Data loading, normalisation and the offline generator."""

import numpy as np
import pandas as pd
import pytest

from quantlab.data import OHLCV, DataError, load_csv, load_pair, load_prices, synthetic_prices


def test_synthetic_is_deterministic():
    a = synthetic_prices("AAPL", periods=200)
    b = synthetic_prices("AAPL", periods=200)
    pd.testing.assert_frame_equal(a, b)


def test_different_symbols_differ():
    a = synthetic_prices("AAPL", periods=200)["Close"]
    b = synthetic_prices("MSFT", periods=200)["Close"]
    assert not np.allclose(a, b)


def test_ohlc_relationships_hold(prices):
    assert (prices["High"] >= prices[["Open", "Close"]].max(axis=1)).all()
    assert (prices["Low"] <= prices[["Open", "Close"]].min(axis=1)).all()
    assert (prices["Close"] > 0).all()
    assert list(prices.columns) == OHLCV


def test_wicks_are_independent(prices):
    """Upper and lower wicks must not be identical.

    If they were, candlestick patterns like the hammer and shooting star
    would be mathematically impossible and those strategies would silently
    never fire.
    """
    upper = prices["High"] - prices[["Open", "Close"]].max(axis=1)
    lower = prices[["Open", "Close"]].min(axis=1) - prices["Low"]
    assert abs(np.corrcoef(upper, lower)[0, 1]) < 0.5


def test_returns_are_plausible(prices):
    """The generator must not manufacture a permanently trending market."""
    r = prices["Close"].pct_change().dropna()
    assert 0.05 < r.std() * np.sqrt(252) < 0.60
    assert abs(r.mean() / r.std() * np.sqrt(252)) < 3.0


def test_index_is_sorted_and_unique(prices):
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique


def test_synthetic_source_needs_no_network():
    df = load_prices("ANY", source="synthetic", start="2022-01-01", end="2022-12-31")
    assert len(df) > 200
    assert df.index.min() >= pd.Timestamp("2022-01-01")
    assert df.index.max() <= pd.Timestamp("2022-12-31")


def test_empty_symbol_rejected():
    with pytest.raises(DataError):
        load_prices("   ")


def test_impossible_range_rejected():
    with pytest.raises(DataError):
        load_prices("AAPL", start="2030-01-01", end="2020-01-01", source="synthetic")


def test_load_pair_aligns(prices):
    a, b = load_pair("AAPL", "MSFT", source="synthetic", start="2021-01-01")
    assert a.index.equals(b.index)


def test_adjusted_close_is_preferred(tmp_path):
    """When Adj Close is present it must replace Close and rescale OHL.

    Trading unadjusted prices invents overnight gaps on every split and
    dividend, which momentum strategies happily trade.
    """
    path = tmp_path / "px.csv"
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "Open": [100.0, 102.0], "High": [101.0, 103.0], "Low": [99.0, 101.0],
        "Close": [100.0, 102.0], "Adj Close": [50.0, 51.0], "Volume": [1e6, 1e6],
    }).to_csv(path, index=False)
    df = load_csv(path)
    assert df["Close"].tolist() == [50.0, 51.0]
    assert df["Open"].iloc[0] == pytest.approx(50.0)


def test_csv_with_bad_rows_is_cleaned(tmp_path):
    path = tmp_path / "messy.csv"
    pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"],
        "Open": [10.0, 11.0, 11.0, np.nan], "High": [10.5, 11.5, 11.5, 12.0],
        "Low": [9.5, 10.5, 10.5, 11.0], "Close": [10.0, 11.0, 11.0, 11.5],
        "Volume": [100, 200, 200, 300],
    }).to_csv(path, index=False)
    df = load_csv(path)
    assert df.index.is_unique  # the duplicate date is collapsed
    assert not df["Close"].isna().any()
