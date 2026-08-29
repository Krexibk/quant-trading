"""Command line interface.

::

    python -m quantlab.cli serve
    python -m quantlab.cli backtest macd --symbol AAPL
    python -m quantlab.cli compare --symbol MSFT
    python -m quantlab.cli optimise macd --symbol AAPL
    python -m quantlab.cli account status
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence

from quantlab import __version__


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install 'quantlab[web]'", file=sys.stderr)
        return 1
    print(f"quantlab {__version__} -> http://{args.host}:{args.port}")
    uvicorn.run("quantlab.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _load(args: argparse.Namespace):
    from quantlab.data import load_prices
    from quantlab.strategies import build_pair_frame, get_strategy

    prices = load_prices(args.symbol, args.start, args.end, args.source)
    label = args.symbol
    if get_strategy(args.strategy).needs_pair:
        if not args.symbol_b:
            # ValueError, not SystemExit: main() catches exceptions to print a
            # clean one-line error, and SystemExit would slip past it and dump
            # a traceback at the user instead.
            raise ValueError(f"{args.strategy} needs --symbol-b")
        prices = build_pair_frame(prices, load_prices(args.symbol_b, args.start, args.end, args.source))
        label = f"{args.symbol}/{args.symbol_b}"
    return prices, label


def cmd_backtest(args: argparse.Namespace) -> int:
    from quantlab.backtest import RiskConfig, run_backtest
    from quantlab.config import CostModel

    prices, label = _load(args)
    params = json.loads(args.params) if args.params else {}
    result = run_backtest(
        args.strategy, prices, params=params, initial_capital=args.capital,
        costs=CostModel(args.commission, args.slippage),
        risk=RiskConfig(
            target_volatility=args.target_vol or None,
            stop_loss_atr=args.stop or None,
        ),
        symbol=label,
    )
    if args.json:
        print(json.dumps(result.to_dict()["stats"], indent=2))
    else:
        print(result.summary())
        if args.trades and result.trades:
            print("\nlast 10 trades")
            for t in result.trades[-10:]:
                print(f"  {t.entry_date} -> {t.exit_date}  {t.direction:5s} "
                      f"{t.entry_price:9.2f} -> {t.exit_price:9.2f}  "
                      f"{t.pnl:+10.2f}  ({t.exit_reason})")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run every strategy on one symbol and rank them."""
    from quantlab.backtest import run_backtest
    from quantlab.data import load_prices
    from quantlab.strategies import list_strategies

    prices = load_prices(args.symbol, args.start, args.end, args.source)
    rows = []
    for strat in list_strategies():
        if strat.needs_pair:
            continue
        try:
            s = run_backtest(strat.name, prices, symbol=args.symbol).stats
            rows.append((strat.name, s))
        except Exception as exc:
            print(f"  {strat.name}: failed ({exc})", file=sys.stderr)

    rows.sort(key=lambda r: r[1].sharpe, reverse=True)
    print(f"\n{args.symbol}  {rows[0][1].start} -> {rows[0][1].end}\n")
    print(f"{'strategy':22s}{'return':>10}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'trades':>8}")
    print("-" * 74)
    for name, s in rows:
        print(f"{name:22s}{_fmt_pct(s.total_return):>10}{_fmt_pct(s.cagr):>9}"
              f"{s.sharpe:>8.2f}{_fmt_pct(s.max_drawdown):>9}{s.calmar:>8.2f}{s.trades:>8d}")
    bench = run_backtest("macd", prices).benchmark_stats
    print("-" * 74)
    print(f"{'buy & hold':22s}{_fmt_pct(bench.total_return):>10}{_fmt_pct(bench.cagr):>9}"
          f"{bench.sharpe:>8.2f}{_fmt_pct(bench.max_drawdown):>9}{bench.calmar:>8.2f}{'-':>8}")
    return 0


def cmd_optimise(args: argparse.Namespace) -> int:
    """Grid-search a strategy's parameters.

    Reports in-sample and out-of-sample results on a chronological split.
    A parameter set that only works in-sample is curve-fitted, and the
    split is the cheapest way to see that.
    """
    from quantlab.backtest import run_backtest
    from quantlab.strategies import get_strategy

    prices, label = _load(args)
    strat = get_strategy(args.strategy)
    split = int(len(prices) * args.split)
    train, test = prices.iloc[:split], prices.iloc[split:]

    grid: dict[str, list] = {}
    for p in strat.params:
        if p.kind == "choice" or p.minimum is None or p.maximum is None:
            continue
        base = float(p.default)
        if base <= 0:
            continue
        lo, hi = max(p.minimum, base * 0.5), min(p.maximum, base * 1.5)
        vals = sorted({round(lo, 4), round(base, 4), round(hi, 4)})
        grid[p.name] = [int(v) if p.kind == "int" else v for v in vals]
    if not grid:
        print("no numeric parameters to optimise", file=sys.stderr)
        return 1

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    if len(combos) > args.max_combos:
        combos = combos[: args.max_combos]
    print(f"testing {len(combos)} parameter sets on {len(train)} train / {len(test)} test bars\n")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo, strict=True))
        try:
            tr = run_backtest(args.strategy, train, params=params, symbol=label).stats
            te = run_backtest(args.strategy, test, params=params, symbol=label).stats
            results.append((params, tr, te))
        except Exception:
            continue
    if not results:
        print("every parameter set failed", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r[1].sharpe, reverse=True)
    print(f"{'parameters':46s}{'IS Sharpe':>11}{'OOS Sharpe':>12}{'OOS return':>12}")
    print("-" * 81)
    for params, tr, te in results[:12]:
        text = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{text[:44]:46s}{tr.sharpe:>11.2f}{te.sharpe:>12.2f}{_fmt_pct(te.total_return):>12}")

    best = results[0]
    print(f"\nbest in-sample: {best[0]}")
    print(f"  in-sample Sharpe {best[1].sharpe:.2f} -> out-of-sample {best[2].sharpe:.2f}")
    if best[2].sharpe < best[1].sharpe * 0.5:
        print("  warning: out-of-sample performance collapsed. This is curve fitting,")
        print("  not an edge. Prefer parameters that hold up on both halves.")
    return 0


def cmd_account(args: argparse.Namespace) -> int:
    from quantlab.banking import Ledger
    from quantlab.data import load_prices

    ledger = Ledger()
    if args.action == "status":
        snap = ledger.portfolio()
        marks = {}
        for pos in snap.positions:
            try:
                marks[pos.symbol] = float(load_prices(pos.symbol).iloc[-1]["Close"])
            except Exception:
                pass
        p = ledger.portfolio(marks)
        print(f"equity        {p.equity:>14,.2f}")
        print(f"cash          {p.cash:>14,.2f}")
        print(f"positions     {p.positions_value:>14,.2f}")
        print(f"net deposits  {p.net_deposits:>14,.2f}")
        print(f"total P&L     {p.total_pnl:>14,.2f}  ({_fmt_pct(p.total_pnl_pct)})")
        if p.positions:
            print(f"\n{'symbol':10s}{'qty':>12}{'avg':>11}{'last':>11}{'value':>13}{'P&L':>12}")
            for pos in p.positions:
                print(f"{pos.symbol:10s}{pos.quantity:>12.4f}{pos.avg_price:>11.2f}"
                      f"{pos.last_price:>11.2f}{pos.market_value:>13,.2f}{pos.unrealised:>+12,.2f}")
    elif args.action == "verify":
        print(json.dumps(ledger.verify_integrity(), indent=2))
    elif args.action == "reset":
        if input("wipe the paper account? type 'yes': ").strip().lower() != "yes":
            print("cancelled")
            return 1
        ledger.reset()
        print("account reset")
    return 0


def cmd_strategies(_: argparse.Namespace) -> int:
    from quantlab.strategies import list_strategies

    for s in list_strategies():
        print(f"\n\033[1m{s.name}\033[0m  ({s.category})")
        print(f"  {s.label}")
        text = s.description
        while text:
            print(f"    {text[:88]}")
            text = text[88:]
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab", description=__doc__)
    parser.add_argument("--version", action="version", version=f"quantlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", default="AAPL")
        p.add_argument("--symbol-b", default=None, help="second leg for pair strategies")
        p.add_argument("--start", default=None)
        p.add_argument("--end", default=None)
        p.add_argument("--source", default="auto",
                       choices=["auto", "yfinance", "cache", "synthetic"])

    s = sub.add_parser("serve", help="run the web UI and API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    b = sub.add_parser("backtest", help="backtest one strategy")
    b.add_argument("strategy")
    common(b)
    b.add_argument("--capital", type=float, default=100_000)
    b.add_argument("--commission", type=float, default=1.0, help="bps")
    b.add_argument("--slippage", type=float, default=2.0, help="bps")
    b.add_argument("--target-vol", type=float, default=0.15, help="0 disables vol targeting")
    b.add_argument("--stop", type=float, default=3.0, help="ATR stop multiple, 0 disables")
    b.add_argument("--params", default=None, help='JSON, e.g. \'{"fast": 8}\'')
    b.add_argument("--trades", action="store_true")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("compare", help="rank every strategy on one symbol")
    common(c)
    c.set_defaults(func=cmd_compare)

    o = sub.add_parser("optimise", help="grid search with an out-of-sample split")
    o.add_argument("strategy")
    common(o)
    o.add_argument("--split", type=float, default=0.7)
    o.add_argument("--max-combos", type=int, default=60)
    o.set_defaults(func=cmd_optimise)

    a = sub.add_parser("account", help="inspect the paper account")
    a.add_argument("action", choices=["status", "verify", "reset"])
    a.set_defaults(func=cmd_account)

    ls = sub.add_parser("strategies", help="list available strategies")
    ls.set_defaults(func=cmd_strategies)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
