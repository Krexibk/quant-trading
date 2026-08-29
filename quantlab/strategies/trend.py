"""Trend and momentum strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import indicators as ind
from quantlab.backtest import RiskConfig
from quantlab.strategies.base import Param, Strategy, register


def _regime_filter(close: pd.Series, window: int) -> pd.Series:
    """+1 where the long-term trend is up, -1 where it is down.

    Trading against the primary trend is the single most reliable way to
    lose money with a momentum system, so most strategies here gate their
    signal on this.
    """
    if window <= 1:
        return pd.Series(1.0, index=close.index)
    trend = ind.sma(close, window)
    return pd.Series(np.where(close >= trend, 1.0, -1.0), index=close.index)


@register
class MACDStrategy(Strategy):
    name = "macd"
    label = "MACD Oscillator"
    category = "momentum"
    description = (
        "Trades the MACD histogram: long while the MACD line is above its signal "
        "line, short below. Unlike the original dual-SMA crossover, this uses "
        "exponential averages and a signal line, and scales exposure by how far "
        "the histogram has stretched relative to its own recent range -- so a "
        "marginal cross gets a small position and a decisive one gets a full one."
    )
    params = [
        Param("fast", "Fast EMA", 12, "int", 2, 100, help="Short EMA span."),
        Param("slow", "Slow EMA", 26, "int", 5, 300, help="Long EMA span."),
        Param("signal", "Signal EMA", 9, "int", 2, 100, help="Span of the signal line."),
        Param("trend_filter", "Regime filter", 200, "int", 0, 400,
              help="Only take longs above this SMA (0 disables, allows shorts)."),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
        Param("scale_window", "Scaling window", 100, "int", 10, 500,
              help="Lookback used to normalise histogram strength."),
    ]

    def compute(self, prices, fast=12, slow=26, signal=9, trend_filter=200,
                allow_short=1, scale_window=100):
        fast, slow = int(fast), int(slow)
        if fast >= slow:
            fast, slow = slow, fast  # a "fast" EMA slower than the slow one is meaningless
        close = prices["Close"]
        m = ind.macd(close, fast, slow, int(signal))

        # Normalise the histogram by its own recent dispersion so the signal
        # is comparable across symbols and price levels.
        scale = m["histogram"].abs().rolling(
            int(scale_window), min_periods=10
        ).quantile(0.8).replace(0.0, np.nan)
        strength = (m["histogram"] / scale).clip(-1.0, 1.0)

        raw = np.sign(m["histogram"]) * strength.abs()
        if int(trend_filter) > 0:
            regime = _regime_filter(close, int(trend_filter))
            raw = raw.where(np.sign(raw) == regime, 0.0)
        if not int(allow_short):
            raw = raw.clip(lower=0.0)

        out = m.copy()
        out["signal"] = raw
        return out


@register
class HeikinAshiStrategy(Strategy):
    name = "heikin_ashi"
    label = "Heikin-Ashi Candlestick"
    category = "momentum"
    description = (
        "Heikin-Ashi candles smooth away the noise that whipsaws a raw-price "
        "momentum system. Goes long on a run of bullish HA candles with no "
        "lower wick (the classic strong-trend pattern) and exits when the "
        "candle body flips or a wick reappears on both sides."
    )
    params = [
        Param("confirm_bars", "Confirmation bars", 2, "int", 1, 10,
              help="Consecutive same-colour candles required to enter."),
        Param("trend_filter", "Regime filter", 100, "int", 0, 400),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def compute(self, prices, confirm_bars=2, trend_filter=100, allow_short=1):
        ha = ind.heikin_ashi(prices)
        bull = ha["ha_close"] > ha["ha_open"]
        bear = ha["ha_close"] < ha["ha_open"]

        # A strong trend candle has no wick against the direction of travel.
        no_lower_wick = np.isclose(ha["ha_low"], ha[["ha_open", "ha_close"]].min(axis=1))
        no_upper_wick = np.isclose(ha["ha_high"], ha[["ha_open", "ha_close"]].max(axis=1))

        k = int(confirm_bars)
        long_ok = bull.rolling(k, min_periods=k).sum() == k
        short_ok = bear.rolling(k, min_periods=k).sum() == k

        raw = pd.Series(0.0, index=prices.index)
        raw[long_ok & no_lower_wick] = 1.0
        raw[long_ok & ~no_lower_wick] = 0.6
        raw[short_ok & no_upper_wick] = -1.0
        raw[short_ok & ~no_upper_wick] = -0.6

        if int(trend_filter) > 0:
            regime = _regime_filter(prices["Close"], int(trend_filter))
            raw = raw.where(np.sign(raw) == regime, 0.0)
        if not int(allow_short):
            raw = raw.clip(lower=0.0)

        out = ha.copy()
        out["signal"] = raw
        return out


@register
class DualThrustStrategy(Strategy):
    name = "dual_thrust"
    label = "Dual Thrust Breakout"
    category = "breakout"
    description = (
        "Opening-range breakout. Builds a band around today's open from the "
        "recent high/low range and goes long or short when price breaks "
        "through it. An ADX gate keeps it out of the sideways markets where "
        "breakout systems bleed on false signals."
    )
    params = [
        Param("lookback", "Range lookback", 4, "int", 2, 60),
        Param("k_up", "Upper multiplier", 0.5, "float", 0.05, 3.0, 0.05),
        Param("k_down", "Lower multiplier", 0.5, "float", 0.05, 3.0, 0.05),
        Param("adx_min", "Min ADX", 20, "int", 0, 60,
              help="Skip signals when trend strength is below this (0 disables)."),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def risk(self) -> RiskConfig:
        # Breakouts need room to breathe but must cut losers fast.
        return RiskConfig(target_volatility=0.15, stop_loss_atr=2.0, trailing_atr=3.5)

    def compute(self, prices, lookback=4, k_up=0.5, k_down=0.5, adx_min=20, allow_short=1):
        n = int(lookback)
        # All components use only *completed* prior bars.
        hh = prices["High"].shift(1).rolling(n, min_periods=n).max()
        lc = prices["Close"].shift(1).rolling(n, min_periods=n).min()
        hc = prices["Close"].shift(1).rolling(n, min_periods=n).max()
        ll = prices["Low"].shift(1).rolling(n, min_periods=n).min()
        rng = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)

        upper = prices["Open"] + float(k_up) * rng
        lower = prices["Open"] - float(k_down) * rng

        raw = pd.Series(0.0, index=prices.index)
        raw[prices["Close"] > upper] = 1.0
        raw[prices["Close"] < lower] = -1.0

        if int(adx_min) > 0:
            strength = ind.adx(prices)["adx"]
            raw = raw.where(strength >= float(adx_min), 0.0)
        if not int(allow_short):
            raw = raw.clip(lower=0.0)

        # Hold the breakout until the opposite side triggers.
        raw = raw.replace(0.0, np.nan).ffill().fillna(0.0)

        return pd.DataFrame(
            {"upper": upper, "lower": lower, "range": rng, "signal": raw}, index=prices.index
        )


@register
class ParabolicSARStrategy(Strategy):
    name = "parabolic_sar"
    label = "Parabolic SAR"
    category = "breakout"
    description = (
        "Stop-and-reverse trend following. The SAR trails price and flips the "
        "position when touched. Adds an ADX gate and an optional regime "
        "filter, because a bare SAR reverses constantly in a range and pays "
        "commission on every flip."
    )
    params = [
        Param("af_step", "AF step", 0.02, "float", 0.005, 0.1, 0.005),
        Param("af_max", "AF maximum", 0.2, "float", 0.05, 1.0, 0.05),
        Param("adx_min", "Min ADX", 25, "int", 0, 60),
        Param("trend_filter", "Regime filter", 0, "int", 0, 400),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def risk(self) -> RiskConfig:
        # The SAR is itself a trailing stop, so don't double up on one.
        return RiskConfig(target_volatility=0.15, stop_loss_atr=4.0, trailing_atr=None)

    def compute(self, prices, af_step=0.02, af_max=0.2, adx_min=25,
                trend_filter=0, allow_short=1):
        sar = ind.parabolic_sar(prices, float(af_step), float(af_max))
        raw = sar["trend"].astype(float)

        if int(adx_min) > 0:
            raw = raw.where(ind.adx(prices)["adx"] >= float(adx_min), 0.0)
        if int(trend_filter) > 0:
            regime = _regime_filter(prices["Close"], int(trend_filter))
            raw = raw.where(np.sign(raw) == regime, 0.0)
        if not int(allow_short):
            raw = raw.clip(lower=0.0)

        out = sar.copy()
        out["signal"] = raw
        return out


@register
class AwesomeOscillatorStrategy(Strategy):
    name = "awesome_oscillator"
    label = "Awesome Oscillator"
    category = "momentum"
    description = (
        "Bill Williams' oscillator: the gap between a fast and slow SMA of the "
        "median price. Trades the zero-line cross plus the 'saucer' pattern "
        "(three bars turning back toward the trend), sized by how extended the "
        "oscillator is."
    )
    params = [
        Param("fast", "Fast SMA", 5, "int", 2, 50),
        Param("slow", "Slow SMA", 34, "int", 5, 200),
        Param("trend_filter", "Regime filter", 100, "int", 0, 400),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def compute(self, prices, fast=5, slow=34, trend_filter=100, allow_short=1):
        fast, slow = int(fast), int(slow)
        if fast >= slow:
            fast, slow = slow, fast
        ao = ind.awesome_oscillator(prices, fast, slow)

        scale = ao.abs().rolling(120, min_periods=20).quantile(0.8).replace(0.0, np.nan)
        raw = (ao / scale).clip(-1.0, 1.0)

        # Saucer: momentum decelerating against the position -> stand down.
        rising = ao.diff() > 0
        raw = raw.where(~((raw > 0) & ~rising & (ao.diff(2) < 0)), raw * 0.5)

        if int(trend_filter) > 0:
            regime = _regime_filter(prices["Close"], int(trend_filter))
            raw = raw.where(np.sign(raw) == regime, 0.0)
        if not int(allow_short):
            raw = raw.clip(lower=0.0)

        return pd.DataFrame({"ao": ao, "signal": raw}, index=prices.index)


@register
class LondonBreakoutStrategy(Strategy):
    name = "london_breakout"
    label = "London Breakout"
    category = "breakout"
    description = (
        "Range breakout in the spirit of the FX session strategy: build a "
        "Donchian channel from the prior N bars and take the break, with an "
        "ATR filter that ignores breaks too small to clear the noise floor. "
        "The original is intraday FX; this is the daily-bar equivalent that "
        "runs on any symbol."
    )
    params = [
        Param("channel", "Channel length", 20, "int", 3, 120),
        Param("exit_channel", "Exit channel", 10, "int", 2, 60,
              help="Opposite-side channel used to exit."),
        Param("atr_filter", "ATR filter", 0.25, "float", 0.0, 3.0, 0.05,
              help="Break must exceed this many ATRs beyond the channel."),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def risk(self) -> RiskConfig:
        return RiskConfig(target_volatility=0.15, stop_loss_atr=2.5, trailing_atr=4.0)

    def compute(self, prices, channel=20, exit_channel=10, atr_filter=0.25, allow_short=1):
        entry = ind.donchian(prices, int(channel))
        exit_ = ind.donchian(prices, int(exit_channel))
        buf = float(atr_filter) * ind.atr(prices)
        close = prices["Close"]

        long_entry = close > entry["upper"] + buf
        short_entry = close < entry["lower"] - buf
        long_exit = close < exit_["lower"]
        short_exit = close > exit_["upper"]

        # Walk the state machine: entries set a position, exits clear it.
        state = np.zeros(len(prices))
        cur = 0.0
        le, se, lx, sx = (s.to_numpy() for s in (long_entry, short_entry, long_exit, short_exit))
        for i in range(len(prices)):
            if cur > 0 and lx[i]:
                cur = 0.0
            elif cur < 0 and sx[i]:
                cur = 0.0
            if cur == 0.0:
                if le[i]:
                    cur = 1.0
                elif se[i]:
                    cur = -1.0
            state[i] = cur

        raw = pd.Series(state, index=prices.index)
        if not int(allow_short):
            raw = raw.clip(lower=0.0)

        return pd.DataFrame(
            {"upper": entry["upper"], "lower": entry["lower"], "signal": raw}, index=prices.index
        )
