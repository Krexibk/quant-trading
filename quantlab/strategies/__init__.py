"""Strategy registry.

Importing this package registers every bundled strategy. Add your own by
subclassing :class:`~quantlab.strategies.base.Strategy` and decorating it with
``@register``.
"""

from __future__ import annotations

import pandas as pd

from quantlab.strategies import mean_reversion, trend  # noqa: F401 (registration)
from quantlab.strategies.base import (
    Param,
    Strategy,
    get_strategy,
    list_strategies,
    register,
    registry,
)

__all__ = [
    "Param", "Strategy", "get_strategy", "list_strategies", "register",
    "registry", "build_pair_frame",
]


def build_pair_frame(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Combine two aligned OHLCV frames into the input the pairs strategy wants.

    Leg A keeps the standard OHLCV names (so the backtester prices the
    position off it) and leg B is carried alongside as ``Close_b``.
    """
    common = a.index.intersection(b.index)
    out = a.loc[common].copy()
    out["Close_b"] = b.loc[common, "Close"]
    return out.dropna()
