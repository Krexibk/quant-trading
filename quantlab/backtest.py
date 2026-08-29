"""The backtest engine.

Design notes, and how this differs from the original scripts:

* **No lookahead.** A signal computed from bar *t*'s close is executed at bar
  *t+1*'s open. The original scripts multiplied a signal by the *same* bar's
  return, which quietly books profits you could not have earned.
* **Real accounting.** Cash and units are tracked separately, so overnight
  gaps, partial rebalances and leverage all behave correctly. Equity is
  ``cash + units * price``, never a cumulative product of a return column.
* **Costs are on by default.** Commission and slippage are charged on the
  traded notional; leverage above 1x pays financing.
* **Stops fill intrabar.** A stop is checked against the bar's low/high and
  fills at the stop price, not at the close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_COSTS, TRADING_DAYS, CostModel
from quantlab.indicators import atr as atr_indicator
from quantlab.metrics import PerformanceStats, drawdown_series, performance_stats


@dataclass
class RiskConfig:
    """Position sizing and stop rules applied on top of a raw signal."""

    #: Scale exposure so realised vol matches this annualised target.
    #: ``None`` disables vol targeting and uses the raw signal as exposure.
    target_volatility: float | None = 0.15
    #: Lookback for the realised-vol estimate.
    vol_lookback: int = 20
    #: Hard cap on absolute exposure, after sizing.
    max_leverage: float = 1.0
    #: Stop loss in ATR multiples from the entry price. ``None`` disables.
    stop_loss_atr: float | None = 3.0
    #: Take profit in ATR multiples. ``None`` disables.
    take_profit_atr: float | None = None
    #: Trailing stop in ATR multiples from the best price since entry.
    trailing_atr: float | None = None
    atr_period: int = 14
    #: Skip rebalances smaller than this fraction of equity, to stop a
    #: continuously-varying target from churning the account in commission.
    rebalance_threshold: float = 0.02


@dataclass
class Trade:
    """A completed round trip."""

    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date, "exit_date": self.exit_date,
            "direction": self.direction, "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4), "units": round(self.units, 6),
            "pnl": round(self.pnl, 2), "return_pct": round(self.return_pct * 100, 3),
            "bars_held": self.bars_held, "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    """Everything the engine produces for one run."""

    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    prices: pd.DataFrame
    signals: pd.DataFrame
    trades: list[Trade]
    stats: PerformanceStats
    benchmark_equity: pd.Series
    benchmark_stats: PerformanceStats
    strategy: str = ""
    symbol: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def drawdown(self) -> pd.Series:
        return drawdown_series(self.equity)

    def summary(self) -> str:
        """A human-readable performance block."""
        s, b = self.stats, self.benchmark_stats
        rows = [
            ("Total return", f"{s.total_return:>10.2%}", f"{b.total_return:>10.2%}"),
            ("CAGR", f"{s.cagr:>10.2%}", f"{b.cagr:>10.2%}"),
            ("Volatility", f"{s.volatility:>10.2%}", f"{b.volatility:>10.2%}"),
            ("Sharpe", f"{s.sharpe:>10.2f}", f"{b.sharpe:>10.2f}"),
            ("Sortino", f"{s.sortino:>10.2f}", f"{b.sortino:>10.2f}"),
            ("Max drawdown", f"{s.max_drawdown:>10.2%}", f"{b.max_drawdown:>10.2%}"),
            ("Calmar", f"{s.calmar:>10.2f}", f"{b.calmar:>10.2f}"),
            ("Hit rate", f"{s.hit_rate:>10.2%}", "         -"),
            ("Profit factor", f"{s.profit_factor:>10.2f}", "         -"),
            ("Exposure", f"{s.exposure:>10.2%}", "         -"),
            ("Trades", f"{s.trades:>10d}", "         -"),
            ("Final equity", f"{s.final_equity:>10,.0f}", f"{b.final_equity:>10,.0f}"),
        ]
        head = f"{self.strategy} on {self.symbol}  [{s.start} -> {s.end}, {s.days} bars]"
        lines = [head, "=" * len(head), f"{'':<16}{'strategy':>12}{'buy & hold':>12}"]
        lines += [f"{name:<16}{a:>12}{c:>12}" for name, a, c in rows]
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            lines.append(
                f"{'Win/loss':<16}{len(wins):>7}/{len(self.trades) - len(wins):<4}"
                f"{'avg hold ' + str(int(np.mean([t.bars_held for t in self.trades]))) + 'd':>12}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-friendly payload for the API and the web UI."""
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "params": self.params,
            "stats": self.stats.to_dict(),
            "benchmark_stats": self.benchmark_stats.to_dict(),
            "dates": [d.strftime("%Y-%m-%d") for d in self.equity.index],
            "equity": [round(v, 2) for v in self.equity.tolist()],
            "benchmark": [round(v, 2) for v in self.benchmark_equity.tolist()],
            "close": [round(v, 4) for v in self.prices["Close"].reindex(self.equity.index).tolist()],
            "position": [round(v, 4) for v in self.position.tolist()],
            "drawdown": [round(v, 5) for v in self.drawdown.tolist()],
            "trades": [t.to_dict() for t in self.trades],
        }


class Backtester:
    """Runs a target-exposure path against prices."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        costs: CostModel | None = None,
        risk: RiskConfig | None = None,
        periods_per_year: int = TRADING_DAYS,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)
        self.costs = costs or DEFAULT_COSTS
        self.risk = risk or RiskConfig()
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------ sizing
    def _size(self, prices: pd.DataFrame, raw: pd.Series) -> pd.Series:
        """Turn a raw signal in [-1, 1] into a target exposure path."""
        raw = raw.reindex(prices.index).fillna(0.0).clip(-1.0, 1.0)
        risk = self.risk
        if risk.target_volatility:
            rets = prices["Close"].pct_change()
            realised = rets.rolling(
                risk.vol_lookback, min_periods=max(2, risk.vol_lookback // 2)
            ).std(ddof=0) * np.sqrt(self.periods_per_year)
            # Floor the estimate: a near-zero vol print would demand
            # astronomical leverage.
            scalar = risk.target_volatility / realised.clip(lower=0.02)
            raw = raw * scalar.fillna(1.0)
        return raw.clip(-risk.max_leverage, risk.max_leverage).fillna(0.0)

    # ------------------------------------------------------------------- engine
    def run(
        self,
        prices: pd.DataFrame,
        raw_signal: pd.Series,
        signals: pd.DataFrame | None = None,
        strategy: str = "",
        symbol: str = "",
        params: dict | None = None,
    ) -> BacktestResult:
        """Simulate ``raw_signal`` on ``prices`` and return a result bundle."""
        prices = prices.dropna(subset=["Open", "High", "Low", "Close"])
        if len(prices) < 2:
            raise ValueError("need at least two bars to backtest")

        target = self._size(prices, raw_signal)
        # Execution lag: a signal seen at the close of t is traded at t+1.
        target = target.shift(1).fillna(0.0)

        atr_vals = atr_indicator(prices, self.risk.atr_period).bfill().to_numpy(float)
        open_ = prices["Open"].to_numpy(float)
        high = prices["High"].to_numpy(float)
        low = prices["Low"].to_numpy(float)
        close = prices["Close"].to_numpy(float)
        tgt = target.to_numpy(float)
        n = len(prices)

        cost_rate = self.costs.per_unit_turnover
        financing_daily = self.costs.financing_bps_pa / 10_000.0 / self.periods_per_year

        cash = self.initial_capital
        units = 0.0
        equity = np.empty(n)
        exposure = np.zeros(n)
        trades: list[Trade] = []

        entry_price = 0.0
        entry_idx = 0
        best_price = 0.0
        stop_px: float | None = None
        # After a stop-out we refuse to re-enter in the same direction until
        # the signal resets (goes flat or flips). Without this lockout the
        # engine re-buys at the very next open on an unchanged signal, which
        # makes a stop loss *increase* drawdown while paying commission on
        # every bounce -- it turns the risk control into a cost centre.
        blocked_direction = 0

        index = prices.index
        r = self.risk

        def _stop_levels(direction: int, anchor: float, ref: float) -> float | None:
            """Worst-case exit price for the current open position."""
            levels: list[float] = []
            if r.stop_loss_atr:
                levels.append(entry_price - direction * r.stop_loss_atr * anchor)
            if r.trailing_atr:
                levels.append(ref - direction * r.trailing_atr * anchor)
            if not levels:
                return None
            return max(levels) if direction > 0 else min(levels)

        for i in range(n):
            px_open = open_[i]
            # --- mark to the open, then rebalance toward the target ----------
            equity_open = cash + units * px_open
            if equity_open <= 0:  # account wiped out; stay flat from here on
                cash, units = max(cash, 0.0), 0.0
                equity[i] = max(cash, 0.0)
                continue

            desired = tgt[i]
            if blocked_direction != 0:
                if np.sign(desired) == blocked_direction:
                    desired = 0.0          # still locked out
                else:
                    blocked_direction = 0  # signal reset -> re-entry allowed

            want_units = desired * equity_open / px_open if px_open > 0 else 0.0
            delta = want_units - units
            # Ignore trivial rebalances so a smoothly-varying target does not
            # bleed the account in commission.
            if abs(delta * px_open) >= r.rebalance_threshold * equity_open or (
                want_units == 0.0 and units != 0.0
            ):
                notional = abs(delta) * px_open
                cash -= delta * px_open + notional * cost_rate
                prev_units = units
                units = want_units

                if prev_units != 0.0 and (units == 0.0 or np.sign(units) != np.sign(prev_units)):
                    direction = 1 if prev_units > 0 else -1
                    pnl = (px_open - entry_price) * prev_units
                    trades.append(
                        Trade(
                            entry_date=str(index[entry_idx])[:10],
                            exit_date=str(index[i])[:10],
                            direction="long" if direction > 0 else "short",
                            entry_price=entry_price, exit_price=px_open,
                            units=prev_units, pnl=pnl,
                            return_pct=direction * (px_open / entry_price - 1.0)
                            if entry_price > 0 else 0.0,
                            bars_held=i - entry_idx, exit_reason="signal",
                        )
                    )
                if units != 0.0 and (prev_units == 0.0 or np.sign(units) != np.sign(prev_units)):
                    entry_price, entry_idx, best_price = px_open, i, px_open
                    stop_px = _stop_levels(1 if units > 0 else -1, atr_vals[i], px_open)

            # --- intrabar risk exit ------------------------------------------
            if units != 0.0:
                direction = 1 if units > 0 else -1
                best_price = max(best_price, high[i]) if direction > 0 else min(best_price, low[i])
                stop_px = _stop_levels(direction, atr_vals[i], best_price)
                target_px = (
                    entry_price + direction * r.take_profit_atr * atr_vals[i]
                    if r.take_profit_atr else None
                )

                exit_px: float | None = None
                reason = ""
                if stop_px is not None and (
                    (direction > 0 and low[i] <= stop_px) or (direction < 0 and high[i] >= stop_px)
                ):
                    # Conservative: a gap through the stop fills at the open.
                    exit_px = min(stop_px, px_open) if direction > 0 else max(stop_px, px_open)
                    reason = "stop"
                elif target_px is not None and (
                    (direction > 0 and high[i] >= target_px)
                    or (direction < 0 and low[i] <= target_px)
                ):
                    exit_px, reason = target_px, "target"

                if exit_px is not None and exit_px > 0:
                    cash += units * exit_px - abs(units) * exit_px * cost_rate
                    trades.append(
                        Trade(
                            entry_date=str(index[entry_idx])[:10], exit_date=str(index[i])[:10],
                            direction="long" if direction > 0 else "short",
                            entry_price=entry_price, exit_price=exit_px, units=units,
                            pnl=(exit_px - entry_price) * units,
                            return_pct=direction * (exit_px / entry_price - 1.0)
                            if entry_price > 0 else 0.0,
                            bars_held=i - entry_idx, exit_reason=reason,
                        )
                    )
                    if reason == "stop":
                        blocked_direction = direction
                    units, stop_px = 0.0, None

            # --- mark to close, charge financing on leverage -----------------
            eq = cash + units * close[i]
            gross = abs(units * close[i])
            if eq > 0 and gross > eq:
                eq -= (gross - eq) * financing_daily
            equity[i] = eq
            exposure[i] = (units * close[i] / eq) if eq > 0 else 0.0

        equity_s = pd.Series(equity, index=index, name="equity")
        position_s = pd.Series(exposure, index=index, name="position")
        returns_s = equity_s.pct_change().fillna(0.0).rename("returns")

        bench = self.initial_capital * prices["Close"] / prices["Close"].iloc[0]
        result_signals = signals if signals is not None else pd.DataFrame(index=index)
        result_signals = result_signals.copy()
        result_signals["raw_signal"] = raw_signal.reindex(index)
        result_signals["target"] = target

        return BacktestResult(
            equity=equity_s,
            returns=returns_s,
            position=position_s,
            prices=prices,
            signals=result_signals,
            trades=trades,
            stats=performance_stats(
                equity_s, returns_s, position_s, self.periods_per_year,
                trade_pnls=[t.pnl for t in trades],
            ),
            benchmark_equity=bench,
            benchmark_stats=performance_stats(
                bench, bench.pct_change(), pd.Series(1.0, index=index), self.periods_per_year
            ),
            strategy=strategy,
            symbol=symbol,
            params=params or {},
        )


def run_backtest(
    strategy: str,
    prices: pd.DataFrame,
    params: dict | None = None,
    initial_capital: float = 100_000.0,
    costs: CostModel | None = None,
    risk: RiskConfig | None = None,
    symbol: str = "",
    **kwargs: Any,
) -> BacktestResult:
    """Look up ``strategy`` by name, generate its signal and backtest it."""
    from quantlab.strategies import get_strategy

    strat = get_strategy(strategy)
    merged = {**strat.defaults, **(params or {}), **kwargs}
    signal_frame = strat.generate(prices, **merged)
    engine = Backtester(initial_capital=initial_capital, costs=costs, risk=risk or strat.risk())
    return engine.run(
        prices,
        signal_frame["signal"],
        signals=signal_frame,
        strategy=strat.name,
        symbol=symbol,
        params=merged,
    )
