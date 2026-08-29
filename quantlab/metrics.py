"""Performance and risk statistics for an equity curve."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantlab.config import TRADING_DAYS


@dataclass
class PerformanceStats:
    """Summary statistics. Rates are decimals (0.12 == 12%)."""

    start: str
    end: str
    days: int
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_days: int
    calmar: float
    hit_rate: float
    profit_factor: float
    best_day: float
    worst_day: float
    exposure: float
    turnover: float
    trades: int
    final_equity: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(value: float, default: float = 0.0) -> float:
    """Collapse NaN/inf to a JSON-serialisable number."""
    return float(value) if np.isfinite(value) else default


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak."""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch, in bars, spent below a previous peak."""
    peak = equity.cummax()
    underwater = equity < peak
    if not underwater.any():
        return 0
    # Number of consecutive underwater bars, reset at each new high.
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max())


def performance_stats(
    equity: pd.Series,
    returns: pd.Series | None = None,
    position: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
    risk_free: float = 0.0,
    trade_pnls: list[float] | None = None,
) -> PerformanceStats:
    """Compute summary statistics for an equity curve.

    Parameters
    ----------
    equity:
        Portfolio value over time, starting at any positive number.
    returns:
        Per-bar strategy returns. Derived from ``equity`` when omitted.
    position:
        Target exposure per bar, used for exposure and turnover.
    risk_free:
        Annualised risk-free rate used in the Sharpe/Sortino numerator.
    trade_pnls:
        Profit of each completed round trip. When supplied, ``hit_rate``,
        ``profit_factor`` and ``trades`` describe *trades* rather than bars.
        Per-bar figures answer a different question ("how often was a day
        green?") and read as misleadingly low for a strategy that is flat
        much of the time.
    """
    equity = pd.Series(equity).dropna().astype(float)
    if len(equity) < 2:
        return PerformanceStats(
            start="", end="", days=len(equity), total_return=0.0, cagr=0.0,
            volatility=0.0, sharpe=0.0, sortino=0.0, max_drawdown=0.0,
            max_drawdown_days=0, calmar=0.0, hit_rate=0.0, profit_factor=0.0,
            best_day=0.0, worst_day=0.0, exposure=0.0, turnover=0.0, trades=0,
            final_equity=float(equity.iloc[-1]) if len(equity) else 0.0,
        )

    rets = (equity.pct_change().dropna() if returns is None else pd.Series(returns).dropna())
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(rets)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = n / periods_per_year
    if years > 0 and equity.iloc[0] > 0 and equity.iloc[-1] > 0:
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    vol = float(rets.std(ddof=0) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    excess = rets - risk_free / periods_per_year
    sharpe = _safe(excess.mean() / rets.std(ddof=0) * np.sqrt(periods_per_year)) if rets.std(ddof=0) > 0 else 0.0

    downside = rets[rets < 0]
    dd_std = downside.std(ddof=0) if len(downside) > 1 else 0.0
    sortino = _safe(excess.mean() / dd_std * np.sqrt(periods_per_year)) if dd_std > 0 else 0.0

    dd = drawdown_series(equity)
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = _safe(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    wins, losses = rets[rets > 0], rets[rets < 0]
    hit_rate = float(len(wins) / n) if n else 0.0
    loss_sum = float(-losses.sum())
    profit_factor = _safe(float(wins.sum()) / loss_sum, 0.0) if loss_sum > 0 else 0.0

    exposure = turnover = 0.0
    trades = 0
    if position is not None:
        pos = pd.Series(position).fillna(0.0).astype(float)
        exposure = float((pos != 0).mean())
        turnover = float(pos.diff().abs().sum())
        sign = np.sign(pos)
        trades = int((sign.diff().fillna(sign.iloc[0] if len(sign) else 0) != 0).sum())

    if trade_pnls is not None:
        trades = len(trade_pnls)
        if trades:
            won = [p for p in trade_pnls if p > 0]
            lost_sum = -sum(p for p in trade_pnls if p < 0)
            hit_rate = len(won) / trades
            profit_factor = _safe(sum(won) / lost_sum, 0.0) if lost_sum > 0 else 0.0
        else:
            hit_rate = profit_factor = 0.0

    return PerformanceStats(
        start=str(equity.index[0])[:10],
        end=str(equity.index[-1])[:10],
        days=n,
        total_return=_safe(total_return),
        cagr=_safe(cagr),
        volatility=_safe(vol),
        sharpe=_safe(sharpe),
        sortino=_safe(sortino),
        max_drawdown=_safe(max_dd),
        max_drawdown_days=max_drawdown_duration(equity),
        calmar=_safe(calmar),
        hit_rate=_safe(hit_rate),
        profit_factor=_safe(profit_factor),
        best_day=_safe(float(rets.max())) if n else 0.0,
        worst_day=_safe(float(rets.min())) if n else 0.0,
        exposure=_safe(exposure),
        turnover=_safe(turnover),
        trades=trades,
        final_equity=_safe(float(equity.iloc[-1])),
    )


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Calendar month returns as a year x month table, in percent."""
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return pd.DataFrame()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100.0
    if monthly.empty:
        return pd.DataFrame()
    table = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.to_numpy()}
    )
    return table.pivot(index="year", columns="month", values="ret").round(2)
