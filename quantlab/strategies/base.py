"""Strategy base class and the name -> strategy registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from quantlab.backtest import RiskConfig


@dataclass
class Param:
    """One tunable parameter, also used to render the web UI form."""

    name: str
    label: str
    default: Any
    kind: str = "int"  # int | float | choice
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    help: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "label": self.label, "default": self.default,
            "kind": self.kind, "min": self.minimum, "max": self.maximum,
            "step": self.step, "choices": self.choices, "help": self.help,
        }


class Strategy:
    """Base class. Subclasses implement :meth:`compute`.

    A strategy returns a *target exposure* in ``[-1, 1]`` per bar, where 1 is
    fully long, -1 fully short and 0 flat. Position sizing, stops and
    execution lag are applied by the backtester, not here -- which keeps
    strategies small and makes them comparable on equal terms.
    """

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[str] = "momentum"
    params: ClassVar[list[Param]] = []
    #: Set on strategies that need a second price series (pairs trading).
    needs_pair: ClassVar[bool] = False

    @property
    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def risk(self) -> RiskConfig:
        """Default risk settings for this strategy. Override to specialise."""
        return RiskConfig()

    def compute(self, prices: pd.DataFrame, **params: Any) -> pd.DataFrame:
        raise NotImplementedError

    def generate(self, prices: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Run :meth:`compute` and sanitise its output.

        Guarantees the caller gets a ``signal`` column that is aligned to the
        price index, free of NaN, and clipped to ``[-1, 1]``.
        """
        merged = {**self.defaults, **params}
        allowed = set(self.defaults)
        merged = {k: v for k, v in merged.items() if k in allowed}
        frame = self.compute(prices, **merged)
        if "signal" not in frame.columns:
            raise ValueError(f"{self.name}: compute() must return a 'signal' column")
        frame = frame.reindex(prices.index)
        frame["signal"] = frame["signal"].astype(float).fillna(0.0).clip(-1.0, 1.0)
        return frame

    def to_dict(self) -> dict:
        return {
            "name": self.name, "label": self.label, "description": self.description,
            "category": self.category, "needs_pair": self.needs_pair,
            "params": [p.to_dict() for p in self.params],
        }


_REGISTRY: dict[str, Strategy] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator that adds a strategy to the registry."""
    instance = cls()
    if not instance.name:
        raise ValueError(f"{cls.__name__} must define a name")
    if instance.name in _REGISTRY:
        raise ValueError(f"duplicate strategy name: {instance.name}")
    _REGISTRY[instance.name] = instance
    return cls


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name, case-insensitively."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def list_strategies() -> list[Strategy]:
    """All registered strategies, ordered by category then name."""
    return sorted(_REGISTRY.values(), key=lambda s: (s.category, s.name))


def registry() -> dict[str, Strategy]:
    return dict(_REGISTRY)
