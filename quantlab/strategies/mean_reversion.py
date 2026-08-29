"""Mean-reversion and statistical-arbitrage strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import indicators as ind
from quantlab.backtest import RiskConfig
from quantlab.strategies.base import Param, Strategy, register


@register
class BollingerStrategy(Strategy):
    name = "bollinger"
    label = "Bollinger Bands Reversion"
    category = "mean-reversion"
    description = (
        "Fades stretched moves: buys when price closes below the lower band "
        "and sells when it closes above the upper one, exiting at the middle "
        "band. Two guards the original lacks -- it stands aside when the bands "
        "are expanding hard (a genuine breakout, not a fade) and when the "
        "long-term trend disagrees, which is when 'buy the dip' becomes "
        "'catch the falling knife'."
    )
    params = [
        Param("window", "Band window", 20, "int", 5, 200),
        Param("num_std", "Std deviations", 2.0, "float", 0.5, 4.0, 0.1),
        Param("exit_at_mid", "Exit at midline", 1, "choice", choices=["0", "1"]),
        Param("trend_filter", "Regime filter", 200, "int", 0, 400,
              help="Block counter-trend fades using this SMA (0 disables)."),
        Param("max_bandwidth_z", "Max bandwidth z", 2.0, "float", 0.0, 5.0, 0.25,
              help="Skip entries when band width is this many z above normal."),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def risk(self) -> RiskConfig:
        # Reversion trades are short-horizon; a wide stop with no trail suits.
        return RiskConfig(target_volatility=0.12, stop_loss_atr=2.5, trailing_atr=None)

    def compute(self, prices, window=20, num_std=2.0, exit_at_mid=1,
                trend_filter=200, max_bandwidth_z=2.0, allow_short=1):
        close = prices["Close"]
        bb = ind.bollinger(close, int(window), float(num_std))

        long_entry = close < bb["lower"]
        short_entry = close > bb["upper"]

        # Volatility-expansion guard: fading an exploding range is how a
        # reversion book gets run over.
        if float(max_bandwidth_z) > 0:
            bw_z = ind.zscore(bb["bandwidth"], max(int(window) * 3, 60))
            calm = (bw_z < float(max_bandwidth_z)) | bw_z.isna()
            long_entry &= calm
            short_entry &= calm

        if int(trend_filter) > 0:
            trend = ind.sma(close, int(trend_filter))
            long_entry &= close >= trend
            short_entry &= close <= trend

        if int(exit_at_mid):
            long_exit = close >= bb["mid"]
            short_exit = close <= bb["mid"]
        else:
            long_exit = close >= bb["upper"]
            short_exit = close <= bb["lower"]

        state = np.zeros(len(prices))
        cur = 0.0
        le, se, lx, sx = (
            s.fillna(False).to_numpy() for s in (long_entry, short_entry, long_exit, short_exit)
        )
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

        out = bb.copy()
        out["signal"] = raw
        return out


@register
class RSIStrategy(Strategy):
    name = "rsi"
    label = "RSI Reversion"
    category = "mean-reversion"
    description = (
        "Buys oversold and sells overbought RSI readings. Entry requires RSI "
        "to cross back *out* of the extreme rather than merely be in it -- "
        "waiting for the turn avoids the classic failure of going long at "
        "RSI 30 and riding it to 10. Optional divergence confirmation."
    )
    params = [
        Param("period", "RSI period", 14, "int", 2, 100),
        Param("oversold", "Oversold level", 30, "int", 5, 49),
        Param("overbought", "Overbought level", 70, "int", 51, 95),
        Param("exit_level", "Exit at RSI", 50, "int", 30, 70),
        Param("trend_filter", "Regime filter", 200, "int", 0, 400),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def risk(self) -> RiskConfig:
        return RiskConfig(target_volatility=0.12, stop_loss_atr=2.5, trailing_atr=None)

    def compute(self, prices, period=14, oversold=30, overbought=70,
                exit_level=50, trend_filter=200, allow_short=1):
        close = prices["Close"]
        r = ind.rsi(close, int(period))
        lo, hi = float(oversold), float(overbought)

        # Cross back out of the extreme zone, not merely inside it.
        long_entry = (r > lo) & (r.shift(1) <= lo)
        short_entry = (r < hi) & (r.shift(1) >= hi)

        if int(trend_filter) > 0:
            trend = ind.sma(close, int(trend_filter))
            long_entry &= close >= trend
            short_entry &= close <= trend

        long_exit = r >= float(exit_level) + (hi - float(exit_level)) * 0.5
        short_exit = r <= float(exit_level) - (float(exit_level) - lo) * 0.5

        state = np.zeros(len(prices))
        cur = 0.0
        le, se, lx, sx = (
            s.fillna(False).to_numpy() for s in (long_entry, short_entry, long_exit, short_exit)
        )
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

        return pd.DataFrame({"rsi": r, "signal": raw}, index=prices.index)


@register
class ShootingStarStrategy(Strategy):
    name = "shooting_star"
    label = "Shooting Star / Hammer"
    category = "mean-reversion"
    description = (
        "Single-candle reversal patterns. A shooting star (long upper wick, "
        "small body, near the top of an up-move) is a short; a hammer at the "
        "bottom of a down-move is a long. Both require the pattern to appear "
        "at a genuine local extreme, which is what separates a reversal "
        "signal from a random small-bodied candle."
    )
    params = [
        Param("wick_ratio", "Min wick/body", 2.0, "float", 1.0, 6.0, 0.25),
        Param("lookback", "Extreme lookback", 10, "int", 3, 60,
              help="Pattern must occur at an N-bar high (star) or low (hammer)."),
        Param("hold_bars", "Max hold", 5, "int", 1, 40),
        Param("allow_short", "Allow shorts", 1, "choice", choices=["0", "1"]),
    ]

    def risk(self) -> RiskConfig:
        return RiskConfig(target_volatility=0.12, stop_loss_atr=2.0, trailing_atr=None)

    def compute(self, prices, wick_ratio=2.0, lookback=10, hold_bars=5, allow_short=1):
        op, hi, lo, cl = (prices[x] for x in ("Open", "High", "Low", "Close"))
        body = (cl - op).abs()
        # A doji has ~zero body; use a floor so the ratio stays finite.
        body_floor = body.replace(0.0, np.nan).fillna((hi - lo).abs() * 0.01 + 1e-9)
        body_top = np.maximum(cl, op)
        body_bottom = np.minimum(cl, op)
        upper_wick = hi - body_top
        lower_wick = body_bottom - lo

        n = int(lookback)
        at_high = hi >= hi.rolling(n, min_periods=n).max()
        at_low = lo <= lo.rolling(n, min_periods=n).min()

        star = (upper_wick >= float(wick_ratio) * body_floor) & (lower_wick <= body_floor) & at_high
        hammer = (lower_wick >= float(wick_ratio) * body_floor) & (upper_wick <= body_floor) & at_low

        raw = pd.Series(0.0, index=prices.index)
        raw[hammer.fillna(False)] = 1.0
        raw[star.fillna(False)] = -1.0
        # Hold the reversal for a fixed window, then stand down.
        raw = raw.replace(0.0, np.nan).ffill(limit=int(hold_bars)).fillna(0.0)

        if not int(allow_short):
            raw = raw.clip(lower=0.0)

        return pd.DataFrame(
            {"upper_wick": upper_wick, "lower_wick": lower_wick, "signal": raw},
            index=prices.index,
        )


@register
class PairsStrategy(Strategy):
    name = "pairs"
    label = "Pair Trading (stat arb)"
    category = "stat-arb"
    needs_pair = True
    description = (
        "Statistical arbitrage on two co-moving assets. Estimates a rolling "
        "hedge ratio, z-scores the spread and fades it. Critically, the hedge "
        "ratio is estimated on a *rolling* window rather than fitted once over "
        "the whole sample -- an in-sample beta is lookahead bias and is the "
        "reason naive pair backtests look spectacular and trade terribly. "
        "Positions are also dropped when the spread stops mean-reverting."
    )
    params = [
        Param("beta_window", "Hedge window", 60, "int", 20, 400,
              help="Rolling window for the hedge ratio."),
        Param("z_window", "Z-score window", 30, "int", 10, 250),
        Param("entry_z", "Entry z", 2.0, "float", 0.5, 5.0, 0.1),
        Param("exit_z", "Exit z", 0.5, "float", 0.0, 3.0, 0.1),
        Param("stop_z", "Stop z", 4.0, "float", 1.0, 10.0, 0.5,
              help="Abandon the trade if the spread diverges this far."),
    ]

    def risk(self) -> RiskConfig:
        # The spread z-score is the risk control; ATR stops on one leg would
        # fire on moves the hedge already offsets.
        return RiskConfig(target_volatility=None, max_leverage=1.0,
                          stop_loss_atr=None, trailing_atr=None)

    def compute(self, prices, beta_window=60, z_window=30, entry_z=2.0,
                exit_z=0.5, stop_z=4.0):
        if "Close_b" not in prices.columns:
            raise ValueError(
                "pairs needs a second series: pass a frame with a 'Close_b' column "
                "(quantlab.strategies.build_pair_frame does this for you)"
            )
        y = np.log(prices["Close"])
        x = np.log(prices["Close_b"])

        beta = ind.rolling_beta(y, x, int(beta_window))
        spread = y - beta * x
        z = ind.zscore(spread, int(z_window))

        state = np.zeros(len(prices))
        cur = 0.0
        zz = z.to_numpy()
        ez, xz, sz = float(entry_z), float(exit_z), float(stop_z)
        for i in range(len(prices)):
            v = zz[i]
            if np.isnan(v):
                cur = 0.0
            else:
                if cur != 0.0 and (abs(v) >= sz or abs(v) <= xz):
                    cur = 0.0  # take profit at the mean, or bail on divergence
                if cur == 0.0:
                    if v <= -ez:
                        cur = 1.0   # spread cheap -> long A, short B
                    elif v >= ez:
                        cur = -1.0
            state[i] = cur

        return pd.DataFrame(
            {"beta": beta, "spread": spread, "zscore": z, "signal": state}, index=prices.index
        )
