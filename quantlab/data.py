"""Market data loading.

Three sources, tried in order and selectable via ``source``:

``yfinance``
    Live download. The original scripts imported ``fix_yahoo_finance``, a
    package that was renamed to ``yfinance`` in 2018 and no longer installs.
``cache``
    A CSV under :data:`quantlab.config.CACHE_DIR`, refreshed once the file is
    older than ``CACHE_TTL_HOURS``.
``synthetic``
    A deterministic offline generator. Seeded from the symbol name, so the
    same symbol always produces the same series. This keeps the test suite,
    CI and the demo UI working with no network and no API key.
"""

from __future__ import annotations

import hashlib
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab import config

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataError(RuntimeError):
    """Raised when prices cannot be produced for a symbol."""


def _cache_file(symbol: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in symbol.upper())
    return config.CACHE_DIR / f"{safe}.csv"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours < config.CACHE_TTL_HOURS


def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce any provider's frame into a clean OHLCV frame.

    Handles yfinance's MultiIndex columns, adjusts naming, drops duplicate
    timestamps and forward-fills the small gaps that holidays create.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) columns even for a single ticker.
        level = 0 if "Close" in df.columns.get_level_values(0) else 1
        df = df.droplevel([i for i in range(df.columns.nlevels) if i != level], axis=1)

    df = df.rename(columns={c: str(c).strip().title() for c in df.columns})
    if "Adj Close" in df.columns and "Close" in df.columns:
        # Prefer split/dividend adjusted prices; an unadjusted series creates
        # fake overnight gaps that momentum strategies happily trade on.
        ratio = (df["Adj Close"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.ffill().fillna(1.0)
        for col in ("Open", "High", "Low"):
            if col in df.columns:
                df[col] = df[col] * ratio
        df["Close"] = df["Adj Close"]

    missing = [c for c in OHLCV if c not in df.columns]
    if "Volume" in missing and "Close" in df.columns:
        df["Volume"] = 0.0
        missing.remove("Volume")
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")

    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.astype(float)
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    # High/Low must actually bracket Open/Close or ATR and breakout logic lie.
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    df.index.name = "Date"
    return df


def _download(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf  # imported lazily: quantlab works without it
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataError("yfinance is not installed") from exc

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or len(raw) == 0:
        raise DataError(f"{symbol}: provider returned no rows")
    return _normalise(raw, symbol)


def synthetic_prices(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    periods: int | None = None,
) -> pd.DataFrame:
    """Generate a deterministic, realistic-looking OHLCV series.

    The process is a geometric random walk with a slow-moving drift and
    stochastic volatility, which produces the trends, chop and volatility
    clusters that make a technical strategy behave the way it would on real
    data. Identical inputs always give identical output.
    """
    seed = int(hashlib.sha256(symbol.upper().encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    if periods is None:
        start_dt = pd.Timestamp(start) if start else end_dt - timedelta(days=5 * 365)
        index = pd.bdate_range(start_dt, end_dt)
    else:
        index = pd.bdate_range(end=end_dt, periods=periods)
    n = len(index)
    if n < 2:
        raise DataError(f"{symbol}: date range too short")

    # Slowly mean-reverting drift => multi-month trends rather than noise.
    # The innovation is deliberately tiny: the drift's steady-state standard
    # deviation is sigma/sqrt(1-phi^2) ~= 3e-4 per day (~7.5% annualised).
    # An earlier, larger value produced permanently trending markets on which
    # every momentum strategy scored a Sharpe above 4 -- realistic-looking
    # charts that silently invalidate every backtest run against them.
    phi, sigma = 0.995, 3.0e-5
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = phi * drift[i - 1] + rng.normal(0, sigma)

    # GARCH-ish volatility so quiet and violent regimes alternate.
    vol = np.empty(n)
    vol[0] = 0.012
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        vol[i] = np.sqrt(
            max(1e-8, 0.000004 + 0.90 * vol[i - 1] ** 2 + 0.05 * (vol[i - 1] * shocks[i - 1]) ** 2)
        )
    vol = np.clip(vol, 0.004, 0.09)

    rets = drift + vol * shocks
    base = 20.0 + (seed % 400)
    close = base * np.exp(np.cumsum(rets))

    prev_close = np.concatenate([[base], close[:-1]])
    open_ = prev_close * (1 + rng.normal(0, 0.25, n) * vol)
    # Draw the two wicks independently. Using one shared span would make the
    # upper and lower wick identical on every bar, which silently makes every
    # candlestick pattern (hammer, shooting star, marubozu) impossible to
    # detect -- the generator would quietly zero out those strategies.
    upper_span = np.abs(rng.normal(0, 1.0, n)) * vol * close * 0.6
    lower_span = np.abs(rng.normal(0, 1.0, n)) * vol * close * 0.6
    high = np.maximum(open_, close) + upper_span
    low = np.minimum(open_, close) - lower_span
    volume = rng.lognormal(mean=13.5, sigma=0.45, size=n) * (1 + 4 * np.abs(rets))

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.round()},
        index=index,
    )
    df.index.name = "Date"
    return _normalise(df, symbol)


def load_prices(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str = "auto",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return an OHLCV frame for ``symbol``.

    Parameters
    ----------
    symbol:
        Ticker understood by the provider, e.g. ``"AAPL"`` or ``"BTC-USD"``.
    start, end:
        ``YYYY-MM-DD`` bounds. Default is the last five years.
    source:
        ``"auto"`` (network, then cache, then synthetic), ``"yfinance"``,
        ``"cache"`` or ``"synthetic"``.
    use_cache:
        Read and write the on-disk CSV cache.
    """
    config.ensure_dirs()
    symbol = symbol.strip().upper()
    if not symbol:
        raise DataError("symbol must not be empty")
    cache = _cache_file(symbol)

    if source == "synthetic":
        return _slice(synthetic_prices(symbol, start, end), start, end)

    if source in {"auto", "cache"} and use_cache and _cache_is_fresh(cache):
        try:
            cached = pd.read_csv(cache, index_col=0, parse_dates=True)
            return _slice(_normalise(cached, symbol), start, end)
        except Exception:
            cache.unlink(missing_ok=True)  # corrupt cache should never be fatal

    if source in {"auto", "yfinance"} and config.ALLOW_NETWORK:
        try:
            df = _download(symbol, start, end)
            if use_cache:
                try:
                    df.to_csv(cache)
                except OSError:
                    pass
            return _slice(df, start, end)
        except Exception as exc:
            if source == "yfinance":
                raise DataError(f"{symbol}: download failed ({exc})") from exc

    # Stale cache beats no data at all.
    if use_cache and cache.exists():
        try:
            cached = pd.read_csv(cache, index_col=0, parse_dates=True)
            return _slice(_normalise(cached, symbol), start, end)
        except Exception:
            pass

    if source == "cache":
        raise DataError(f"{symbol}: nothing in the cache")
    return _slice(synthetic_prices(symbol, start, end), start, end)


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        raise DataError("no rows in the requested date range")
    return df


def load_pair(
    symbol_a: str,
    symbol_b: str,
    start: str | None = None,
    end: str | None = None,
    source: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load two symbols aligned on their common trading days."""
    a = load_prices(symbol_a, start, end, source)
    b = load_prices(symbol_b, start, end, source)
    common = a.index.intersection(b.index)
    if len(common) < 30:
        raise DataError(f"{symbol_a}/{symbol_b}: only {len(common)} overlapping days")
    return a.loc[common], b.loc[common]


def load_csv(path: str | Path, symbol: str = "CSV") -> pd.DataFrame:
    """Load a local OHLCV CSV, e.g. the data files shipped in this repo."""
    df = pd.read_csv(path)
    date_col = next(
        (c for c in df.columns if str(c).strip().lower() in {"date", "datetime", "time"}),
        df.columns[0],
    )
    df = df.set_index(date_col)
    return _normalise(df, symbol)
