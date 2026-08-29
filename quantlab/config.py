"""Central configuration for quantlab.

Every value can be overridden with an environment variable so that the same
code runs unchanged on a laptop, in CI and in a container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


HOME = _env_path("QUANTLAB_HOME", Path.home() / ".quantlab")
CACHE_DIR = _env_path("QUANTLAB_CACHE", HOME / "cache")
DB_PATH = _env_path("QUANTLAB_DB", HOME / "quantlab.db")

#: Allow network calls to the market data provider. Set to 0 to force the
#: deterministic offline generator (useful in CI and on planes).
ALLOW_NETWORK = _env_bool("QUANTLAB_ALLOW_NETWORK", True)

#: How long a cached price file stays fresh, in hours.
CACHE_TTL_HOURS = _env_float("QUANTLAB_CACHE_TTL_HOURS", 12.0)


@dataclass(frozen=True)
class CostModel:
    """Round-trip trading frictions, expressed in basis points of notional.

    The original scripts in this repository assumed frictionless trading --
    "no slippage, no surcharge, no illiquidity". That assumption flatters
    every high-turnover strategy, so costs are on by default here.
    """

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    #: Annualised financing cost applied to leverage above 1x.
    financing_bps_pa: float = 150.0

    @property
    def per_unit_turnover(self) -> float:
        """Cost charged per 1.0 of absolute position change."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0


DEFAULT_COSTS = CostModel(
    commission_bps=_env_float("QUANTLAB_COMMISSION_BPS", 1.0),
    slippage_bps=_env_float("QUANTLAB_SLIPPAGE_BPS", 2.0),
)

#: Trading days per year, used to annualise returns and volatility.
TRADING_DAYS = 252


def ensure_dirs() -> None:
    """Create the on-disk directories quantlab writes to."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
