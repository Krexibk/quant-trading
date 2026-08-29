# quantlab — how to use it

quantlab turns the loose collection of backtest scripts in this repository into
something you can actually run: an installable package, a tested backtest
engine, nine improved strategies, a paper-trading account, and a web UI.

---

## 1. Install

```bash
git clone https://github.com/wisk-inc/quant-trading.git
cd quant-trading
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

Python 3.10 or newer. The only hard dependencies are pandas and numpy; the web
UI needs `fastapi` + `uvicorn`, and live market data needs `yfinance`.

Verify:

```bash
quantlab --version
pytest -q          # 178 tests
```

### Works offline

If `yfinance` is missing, the network is unavailable, or the provider is down,
quantlab falls back to a deterministic price generator instead of crashing.
Prices are seeded from the symbol name, so `AAPL` always produces the same
series. Force it with `--source synthetic` or `QUANTLAB_ALLOW_NETWORK=0`.

Synthetic prices are **simulated, not real**. Use them to exercise the machinery
and to write tests — never to decide that a strategy works.

---

## 2. The web UI

```bash
quantlab serve                 # http://127.0.0.1:8000
quantlab serve --port 9000 --reload
```

Five views:

| View | What it does |
|---|---|
| **Dashboard** | Equity, P&L, holdings and allocation. |
| **Funding** | Link a funding source, deposit and withdraw. |
| **Trade** | Live quote, price chart, market/limit orders with a cost preview. |
| **Backtest** | Run any strategy, tune parameters, compare against buy & hold. |
| **Activity** | Full order and transfer history, plus a ledger integrity check. |

The UI has no build step and no external requests — the charts are drawn by
hand on a canvas, so it works with the network completely off.

### Adding funds

1. **Funding → Link a funding source.** Enter a nickname, institution, account
   type and **the last four digits only**.
2. **Move money.** Enter an amount and hit Deposit. Sandbox transfers settle
   instantly.
3. **Trade → place an order.** The server prices every fill.

> **What this will not accept, and why.** quantlab refuses to store bank
> credentials, full account numbers, routing numbers, IBANs or card numbers. The
> server rejects them outright rather than truncating, because truncating would
> mean the number reached the server in the first place. This is deliberate:
> handling that data puts you in PCI-DSS and banking-regulation scope, and a
> local SQLite file is not a safe place for it.
>
> **All money in quantlab is fictional.** See §6 to connect real money properly.

---

## 3. The command line

```bash
# One strategy
quantlab backtest macd --symbol AAPL --start 2019-01-01
quantlab backtest bollinger --symbol TSLA --params '{"window": 30, "num_std": 2.5}' --trades

# Rank every strategy on a symbol
quantlab compare --symbol NVDA --start 2020-01-01

# Grid search with an out-of-sample split
quantlab optimise rsi --symbol SPY --split 0.7

# Pairs trading needs two legs
quantlab backtest pairs --symbol KO --symbol-b PEP

# The paper account
quantlab account status
quantlab account verify      # re-derives cash from the transaction log
quantlab strategies          # descriptions of all nine
```

Useful flags: `--source synthetic|yfinance|cache|auto`, `--commission`/`--slippage`
(basis points), `--target-vol`, `--stop` (ATR multiples), `--json`.

`optimise` reports in-sample **and** out-of-sample Sharpe, and warns when
out-of-sample performance collapses. A parameter set that only works on the
data you tuned it on is curve fitting, not an edge.

---

## 4. The Python API

```python
from quantlab import load_prices, run_backtest

prices = load_prices("AAPL", start="2019-01-01")
result = run_backtest("macd", prices, symbol="AAPL")

print(result.summary())
print(result.stats.sharpe, result.stats.max_drawdown)
result.equity.plot()
```

Tune costs and risk explicitly:

```python
from quantlab.backtest import Backtester, RiskConfig, run_backtest
from quantlab.config import CostModel

result = run_backtest(
    "bollinger", prices,
    params={"window": 30, "num_std": 2.5},
    costs=CostModel(commission_bps=2.0, slippage_bps=5.0),
    risk=RiskConfig(target_volatility=0.10, stop_loss_atr=2.5, max_leverage=1.0),
)
```

Drive the paper account:

```python
from quantlab.banking import Ledger

led = Ledger()
led.link_account("Main", "Chase", "checking", last4="4321")
led.deposit(50_000)
led.place_order("AAPL", "buy", quantity=100, price=190.0)
print(led.portfolio({"AAPL": 205.0}).to_dict())
assert led.verify_integrity()["ok"]
```

### Writing your own strategy

```python
import pandas as pd
from quantlab import indicators as ind
from quantlab.strategies import Param, Strategy, register

@register
class MyStrategy(Strategy):
    name = "my_strategy"
    label = "My Strategy"
    category = "momentum"
    description = "Long when price is above its N-day average."
    params = [Param("window", "Lookback", 50, "int", 5, 300)]

    def compute(self, prices, window=50):
        trend = ind.sma(prices["Close"], int(window))
        signal = (prices["Close"] > trend).astype(float)
        return pd.DataFrame({"trend": trend, "signal": signal})
```

Return a **target exposure** in `[-1, 1]`: 1 fully long, -1 fully short, 0 flat.
Execution lag, sizing, stops and costs are the backtester's job — which is what
keeps every strategy comparable on equal terms. Import the module once and it
appears in the CLI, the API and the web UI automatically.

---

## 5. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `QUANTLAB_HOME` | `~/.quantlab` | Base directory |
| `QUANTLAB_DB` | `$HOME/quantlab.db` | Paper-account database |
| `QUANTLAB_CACHE` | `$HOME/cache` | Price cache |
| `QUANTLAB_ALLOW_NETWORK` | `1` | `0` forces offline mode |
| `QUANTLAB_CACHE_TTL_HOURS` | `12` | Cache freshness |
| `QUANTLAB_COMMISSION_BPS` | `1.0` | Default commission |
| `QUANTLAB_SLIPPAGE_BPS` | `2.0` | Default slippage |

---

## 6. Connecting real money

quantlab moves fictional money. To move real money, implement the
`FundingProvider` protocol in `quantlab/banking/providers.py` against a service
that holds the sensitive data for you:

- **[Plaid](https://plaid.com/docs/link/)** — the user authenticates with their
  bank inside Plaid Link; your server sees only an opaque `access_token` and the
  last four digits.
- **[Stripe](https://docs.stripe.com/payments/quickstart)** — card and ACH
  details are collected by Stripe Elements in an iframe your JavaScript cannot
  read; you receive a `PaymentMethod` id.
- **[Alpaca](https://docs.alpaca.markets/)** — a real brokerage with a free
  paper-trading environment; it handles funding, custody and execution.

```python
class MyProvider:
    name = "alpaca"
    def link_account(self, institution, account_type, last4) -> str: ...
    def deposit(self, reference, amount, currency) -> TransferResult: ...
    def withdraw(self, reference, amount, currency) -> TransferResult: ...

Ledger(provider=MyProvider())
```

Nothing else in the codebase changes.

Before you route real money through anything here, be clear-eyed: **these are
public technical-analysis strategies.** On the offline test data every one of
them scores a Sharpe near zero after costs, which is exactly what published
strategies should do. Backtest results are not predictions, and a strategy
tuned until it looks good on history is the most expensive mistake in the field.

---

## 7. What was wrong with the original scripts

The scripts under `quant-strategies/` are kept as-is for reference. They do not
run correctly on a current Python stack:

1. **Signals were silently all zeros.** Every script assigns positions with
   chained indexing (`signals['positions'][ma1:] = ...`), 94 occurrences across
   the collection. Under pandas' copy-on-write — the default since 3.0 — this
   writes to a temporary copy and is discarded. The script still runs, still
   plots, and reports a flat equity curve built from no positions at all.
2. **`fix_yahoo_finance` no longer exists.** Renamed to `yfinance` in 2018;
   7 files still import the old name and cannot start.
3. **Lookahead bias.** Signals were applied to the same bar's return, booking
   profits that were not available at the time of the decision.
4. **Frictionless by assumption.** The README says "no slippage, no surcharge,
   no illiquidity", which flatters every high-turnover strategy.
5. **In-sample hedge ratios.** The pair-trading script fits one regression over
   the whole sample, then trades the residual — using the whole sample's
   information at every point in it.
6. **No tests, no packaging, no reuse** — global mutable state and hard-coded
   parameters at module level.

`quantlab/` fixes all six. See §8 for how the strategies themselves changed.

---

## 8. What changed in the strategies

Every strategy now emits a target exposure and inherits the same execution
model. Beyond that:

| Strategy | Improvement |
|---|---|
| **MACD** | Was a dual-SMA crossover mislabelled as MACD. Now uses real EMAs and a signal line, and scales exposure by histogram strength relative to its own recent range. |
| **Bollinger** | Stands aside when band width is expanding hard (a breakout, not a fade) and when the long-term trend disagrees — the difference between "buy the dip" and catching a falling knife. |
| **RSI** | Requires RSI to cross back *out* of the extreme rather than merely be in it, avoiding the classic long at RSI 30 that rides to 10. |
| **Parabolic SAR** | Correct SAR clamping against the prior two bars' extremes, plus an ADX gate so it stops reversing on every wiggle in a range. |
| **Dual Thrust** | Range built strictly from completed prior bars; ADX gate; ATR stop and trailing stop. |
| **Heikin-Ashi** | Vectorised (the original looped with `.iloc` assignment); wick-based strength grading; regime filter. |
| **Pairs** | Rolling hedge ratio instead of one in-sample regression, plus a divergence stop. |
| **London Breakout** | Generalised to a Donchian breakout with an ATR noise filter and a separate exit channel. |
| **Awesome Oscillator** | Normalised by its own dispersion; saucer deceleration reduces size. |
| **Shooting Star** | Patterns must occur at a genuine N-bar extreme, which is what separates a reversal from a random small-bodied candle. |

Risk controls applied to all of them: volatility targeting, a leverage cap,
ATR stop-losses that fill intrabar at the stop price, optional trailing stops,
a rebalance threshold to stop commission churn, and a **stop lockout** that
prevents re-entry in the same direction until the signal resets.

That last one is not cosmetic. Without it, a stopped-out position re-enters at
the very next open on an unchanged signal — in testing, that made a 2×ATR stop
*increase* max drawdown from 22.9% to 24.2% while paying commission on every
bounce. With the lockout the same stop cuts drawdown to 6.7%.

---

## 9. Testing

```bash
pytest -q                                   # 178 tests
pytest --cov=quantlab --cov-report=term     # ~91% coverage
```

The suite is fully offline and deterministic. The tests worth knowing about:

- `test_same_bar_signal_is_not_profitable` — trading the sign of *today's*
  return must lose money. If it ever passes, lookahead has crept back in.
- `test_perfect_foresight_is_profitable` — the mirror image, so the test above
  cannot pass for the wrong reason.
- `test_no_lookahead` — an indicator's value at time *t* must not change when
  future bars are appended.
- `test_pairs_hedge_ratio_is_rolling` — the hedge ratio recomputed on truncated
  history must match, proving it is not fitted in-sample.
- `test_integrity_survives_fractional_quantities` — cash re-derived from the
  transaction log must match the stored balance to the cent.
- `test_survives_flat_prices` — a constant price series divides by zero in
  every volatility-based indicator; strategies must return clean signals anyway.
