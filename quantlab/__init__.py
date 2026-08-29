"""quantlab - a backtesting engine, paper-trading ledger and web UI for the
quant-trading strategy collection.

Quick start::

    from quantlab import load_prices, run_backtest
    prices = load_prices("AAPL")
    result = run_backtest("macd", prices)
    print(result.stats)
"""

from quantlab.backtest import Backtester, BacktestResult, run_backtest
from quantlab.data import load_prices
from quantlab.metrics import performance_stats
from quantlab.strategies import get_strategy, list_strategies

__version__ = "1.0.0"

__all__ = [
    "BacktestResult",
    "Backtester",
    "run_backtest",
    "load_prices",
    "performance_stats",
    "get_strategy",
    "list_strategies",
    "__version__",
]
