"""Vectorised technical indicators.

Every function takes and returns pandas objects aligned to the input index,
and none of them look into the future: a value at time *t* uses only data up
to and including *t*. Shifting for execution is the backtester's job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "wilder_ema", "macd", "rsi", "bollinger", "true_range",
    "atr", "adx", "stochastic", "heikin_ashi", "parabolic_sar", "donchian",
    "awesome_oscillator", "zscore", "rolling_beta", "realised_vol",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (``adjust=False``, the trading convention)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, used by RSI/ATR/ADX (alpha = 1/period)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram.

    The repository's original script compared two *simple* moving averages and
    called the difference an oscillator. That is a dual-SMA crossover, not a
    MACD: the real indicator uses exponential averages and a signal line.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}
    )


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, bounded to [0, 100]."""
    delta = close.diff()
    gain = wilder_ema(delta.clip(lower=0.0), period)
    loss = wilder_ema((-delta).clip(lower=0.0), period)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # loss == 0 means an unbroken run of gains -> RSI is 100 by definition.
    return out.where(loss != 0.0, 100.0).where(gain.notna())


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands plus %B and bandwidth."""
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = mid + num_std * sd, mid - num_std * sd
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "pct_b": (close - lower) / width,
            "bandwidth": (upper - lower) / mid.replace(0.0, np.nan),
        }
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range: the largest of today's span and the two gap distances."""
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range (Wilder-smoothed)."""
    return wilder_ema(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Directional movement: +DI, -DI and ADX -- a trend-strength filter."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr_ = wilder_ema(true_range(df), period).replace(0.0, np.nan)
    plus_di = 100.0 * wilder_ema(plus_dm, period) / atr_
    minus_di = 100.0 * wilder_ema(minus_dm, period) / atr_
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": wilder_ema(dx, period)})


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic oscillator %K and %D."""
    low = df["Low"].rolling(k_period, min_periods=k_period).min()
    high = df["High"].rolling(k_period, min_periods=k_period).max()
    span = (high - low).replace(0.0, np.nan)
    k = 100.0 * (df["Close"] - low) / span
    return pd.DataFrame({"k": k, "d": k.rolling(d_period, min_periods=d_period).mean()})


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi candles.

    HA open is recursive (it depends on the previous HA candle), which is why
    this one loop cannot be vectorised away. It runs on numpy arrays rather
    than ``.iloc`` assignment, which is roughly two orders of magnitude
    faster than the row-by-row version in the original script.
    """
    op, hi, lo, cl = (df[col].to_numpy(float) for col in ("Open", "High", "Low", "Close"))
    ha_close = (op + hi + lo + cl) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (op[0] + cl[0]) / 2.0
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": np.maximum.reduce([hi, ha_open, ha_close]),
            "ha_low": np.minimum.reduce([lo, ha_open, ha_close]),
            "ha_close": ha_close,
        },
        index=df.index,
    )


def parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """Parabolic SAR with the standard acceleration factor schedule.

    Returns the stop level and the prevailing trend (+1 long, -1 short). The
    SAR is clamped so it never penetrates the prior two bars' extremes, the
    rule the original implementation left out.
    """
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    n = len(df)
    sar = np.empty(n)
    trend = np.ones(n, dtype=int)

    if n == 0:
        return pd.DataFrame({"sar": [], "trend": []}, index=df.index)

    sar[0] = low[0]
    ep = high[0]
    af = af_step
    for i in range(1, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if trend[i - 1] > 0:
            cur = min(cur, low[i - 1], low[max(0, i - 2)])
            if low[i] < cur:  # stop hit -> flip short
                trend[i], cur, ep, af = -1, ep, low[i], af_step
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep, af = high[i], min(af + af_step, af_max)
        else:
            cur = max(cur, high[i - 1], high[max(0, i - 2)])
            if high[i] > cur:  # stop hit -> flip long
                trend[i], cur, ep, af = 1, ep, high[i], af_step
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep, af = low[i], min(af + af_step, af_max)
        sar[i] = cur
    return pd.DataFrame({"sar": sar, "trend": trend}, index=df.index)


def donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Donchian channel over the *previous* ``window`` bars.

    The shift matters: including today's own high in "the highest high" makes
    a breakout test trivially true and is a classic lookahead bug.
    """
    upper = df["High"].shift(1).rolling(window, min_periods=window).max()
    lower = df["Low"].shift(1).rolling(window, min_periods=window).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


def awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    """Bill Williams' Awesome Oscillator on the median price."""
    median_price = (df["High"] + df["Low"]) / 2.0
    return sma(median_price, fast) - sma(median_price, slow)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score, the workhorse of mean-reversion signals."""
    mean = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / sd


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS hedge ratio of ``y`` on ``x``."""
    cov = y.rolling(window, min_periods=window).cov(x)
    var = x.rolling(window, min_periods=window).var(ddof=0).replace(0.0, np.nan)
    return cov / var


def realised_vol(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualised rolling standard deviation of returns."""
    return returns.rolling(window, min_periods=max(2, window // 2)).std(ddof=0) * np.sqrt(
        periods_per_year
    )
