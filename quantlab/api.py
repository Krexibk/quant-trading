"""FastAPI backend.

Run it with::

    uvicorn quantlab.api:app --reload

or ``python -m quantlab.cli serve``. The web UI is served from ``/``.

Security note: the order endpoint resolves the execution price on the server
rather than trusting a price sent by the browser. A client that could name
its own fill price could mint money in the ledger.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantlab import __version__, config
from quantlab.backtest import RiskConfig, run_backtest
from quantlab.banking import InsufficientFunds, Ledger, LedgerError, UnsafeFieldError
from quantlab.config import CostModel
from quantlab.data import DataError, load_prices
from quantlab.strategies import build_pair_frame, get_strategy, list_strategies

log = logging.getLogger("quantlab.api")

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(
    title="quantlab",
    version=__version__,
    description="Backtesting engine and paper-trading account for the quant-trading strategies.",
)

# The UI is served from the same origin; CORS is here only so you can point a
# separate dev server at the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ledger: Ledger | None = None


def get_ledger() -> Ledger:
    """Lazily open the ledger so importing this module never touches disk."""
    global _ledger
    if _ledger is None:
        _ledger = Ledger()
    return _ledger


def set_ledger(ledger: Ledger) -> None:
    """Swap in a ledger. Used by the test suite to stay off the real database."""
    global _ledger
    _ledger = ledger


# --------------------------------------------------------------------- schemas
class LinkAccountRequest(BaseModel):
    """Only non-sensitive metadata. See quantlab/banking/providers.py."""

    # Reject unknown fields outright. Pydantic would otherwise drop a stray
    # `routing_number` silently -- safe, but it leaves the caller believing
    # the value was accepted. A loud 422 tells them to stop sending it.
    model_config = ConfigDict(extra="forbid")

    nickname: str = Field(..., min_length=1, max_length=60)
    institution: str = Field(..., min_length=1, max_length=80)
    account_type: Literal["checking", "savings", "brokerage", "card"] = "checking"
    last4: str = Field(..., min_length=1, max_length=8, description="Last four digits only")
    currency: str = Field("USD", min_length=3, max_length=3)


class TransferRequest(BaseModel):
    amount: float = Field(..., gt=0, le=1_000_000)
    account_id: str | None = None


class OrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0, le=1_000_000)
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = Field(None, gt=0)
    strategy: str | None = None

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class BacktestRequest(BaseModel):
    strategy: str
    symbol: str = "AAPL"
    symbol_b: str | None = Field(None, description="Second leg, for pair strategies")
    start: str | None = None
    end: str | None = None
    capital: float = Field(100_000, gt=0, le=1e12)
    params: dict[str, Any] = Field(default_factory=dict)
    commission_bps: float = Field(1.0, ge=0, le=500)
    slippage_bps: float = Field(2.0, ge=0, le=500)
    target_volatility: float | None = Field(0.15, ge=0, le=2.0)
    max_leverage: float = Field(1.0, gt=0, le=5.0)
    stop_loss_atr: float | None = Field(3.0, ge=0, le=20)
    trailing_atr: float | None = Field(None, ge=0, le=20)
    source: Literal["auto", "yfinance", "cache", "synthetic"] = "auto"

    @field_validator("symbol", "symbol_b")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else v


# ------------------------------------------------------------------ exceptions
@app.exception_handler(LedgerError)
async def _ledger_error(_, exc: LedgerError) -> JSONResponse:
    status = 402 if isinstance(exc, InsufficientFunds) else 409
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.exception_handler(UnsafeFieldError)
async def _unsafe_field(_, exc: UnsafeFieldError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(DataError)
async def _data_error(_, exc: DataError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# --------------------------------------------------------------------- market
@lru_cache(maxsize=64)
def _cached_prices(symbol: str, start: str | None, end: str | None, source: str, bucket: str):
    """Cache price frames for a few minutes.

    ``bucket`` is a coarse timestamp: including it in the key makes the cache
    expire naturally without a background task.
    """
    return load_prices(symbol, start, end, source)


def _prices(symbol: str, start=None, end=None, source="auto"):
    bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")[:-1]  # ~10 minutes
    return _cached_prices(symbol, start, end, source, bucket)


def last_price(symbol: str, source: str = "auto") -> float:
    """Most recent close for ``symbol``."""
    return float(_prices(symbol, source=source)["Close"].iloc[-1])


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "network_allowed": config.ALLOW_NETWORK,
        "database": str(config.DB_PATH),
        "paper_trading": True,
    }


@app.get("/api/strategies")
def strategies() -> dict:
    return {"strategies": [s.to_dict() for s in list_strategies()]}


@app.get("/api/quote/{symbol}")
def quote(symbol: str, source: str = "auto") -> dict:
    df = _prices(symbol.upper(), source=source)
    close = df["Close"]
    prev = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
    price = float(close.iloc[-1])
    return {
        "symbol": symbol.upper(),
        "price": round(price, 4),
        "previous_close": round(prev, 4),
        "change": round(price - prev, 4),
        "change_pct": round((price / prev - 1) * 100, 3) if prev else 0.0,
        "as_of": str(df.index[-1])[:10],
        "synthetic": source == "synthetic" or not config.ALLOW_NETWORK,
    }


@app.get("/api/history/{symbol}")
def history(symbol: str, days: int = Query(180, ge=5, le=5000), source: str = "auto") -> dict:
    df = _prices(symbol.upper(), source=source).tail(days)
    return {
        "symbol": symbol.upper(),
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "close": [round(v, 4) for v in df["Close"].tolist()],
        "volume": [int(v) for v in df["Volume"].tolist()],
    }


# -------------------------------------------------------------------- backtest
@app.post("/api/backtest")
def backtest(req: BacktestRequest) -> dict:
    try:
        strat = get_strategy(req.strategy)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    prices = _prices(req.symbol, req.start, req.end, req.source)
    symbol_label = req.symbol
    if strat.needs_pair:
        if not req.symbol_b:
            raise HTTPException(
                status_code=400, detail=f"{strat.label} needs a second symbol (symbol_b)"
            )
        other = _prices(req.symbol_b, req.start, req.end, req.source)
        prices = build_pair_frame(prices, other)
        symbol_label = f"{req.symbol}/{req.symbol_b}"
        if len(prices) < 60:
            raise HTTPException(status_code=400, detail="not enough overlapping history")

    if len(prices) < 30:
        raise HTTPException(status_code=400, detail="need at least 30 bars to backtest")

    risk = RiskConfig(
        target_volatility=req.target_volatility or None,
        max_leverage=req.max_leverage,
        stop_loss_atr=req.stop_loss_atr or None,
        trailing_atr=req.trailing_atr or None,
    )
    result = run_backtest(
        req.strategy, prices, params=req.params, initial_capital=req.capital,
        costs=CostModel(req.commission_bps, req.slippage_bps),
        risk=risk, symbol=symbol_label,
    )
    return result.to_dict()


# --------------------------------------------------------------------- funding
@app.get("/api/funding/accounts")
def list_accounts() -> dict:
    return {"accounts": get_ledger().list_accounts()}


@app.post("/api/funding/accounts", status_code=201)
def link_account(req: LinkAccountRequest) -> dict:
    return get_ledger().link_account(
        nickname=req.nickname, institution=req.institution,
        account_type=req.account_type, last4=req.last4, currency=req.currency,
    )


@app.delete("/api/funding/accounts/{account_id}")
def remove_account(account_id: str) -> dict:
    get_ledger().remove_account(account_id)
    return {"deleted": account_id}


@app.post("/api/funding/deposit")
def deposit(req: TransferRequest) -> dict:
    return get_ledger().deposit(req.amount, req.account_id)


@app.post("/api/funding/withdraw")
def withdraw(req: TransferRequest) -> dict:
    return get_ledger().withdraw(req.amount, req.account_id)


@app.get("/api/funding/transfers")
def transfers(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"transfers": get_ledger().list_transfers(limit)}


@app.post("/api/funding/transfers/{transfer_id}/settle")
def settle(transfer_id: str) -> dict:
    return get_ledger().settle_transfer(transfer_id)


# ------------------------------------------------------------------- portfolio
@app.get("/api/portfolio")
def portfolio(source: str = "auto") -> dict:
    ledger = get_ledger()
    snapshot = ledger.portfolio()
    marks: dict[str, float] = {}
    for pos in snapshot.positions:
        try:
            marks[pos.symbol] = last_price(pos.symbol, source)
        except Exception as exc:  # a dead ticker must not break the dashboard
            log.warning("could not price %s: %s", pos.symbol, exc)
    return ledger.portfolio(marks).to_dict()


@app.get("/api/orders")
def orders(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"orders": get_ledger().list_orders(limit)}


@app.post("/api/orders", status_code=201)
def place_order(req: OrderRequest, source: str = "auto") -> dict:
    # The server prices the order. Never trust a price from the client.
    price = last_price(req.symbol, source)
    return get_ledger().place_order(
        symbol=req.symbol, side=req.side, quantity=req.quantity, price=price,
        order_type=req.order_type, limit_price=req.limit_price, strategy=req.strategy,
    )


@app.post("/api/orders/{order_id}/cancel")
def cancel_order(order_id: str) -> dict:
    return get_ledger().cancel_order(order_id)


@app.get("/api/integrity")
def integrity() -> dict:
    return get_ledger().verify_integrity()


@app.post("/api/admin/reset")
def reset(confirm: bool = Body(False, embed=True)) -> dict:
    """Wipe the paper account. Requires ``{"confirm": true}``."""
    if not confirm:
        raise HTTPException(status_code=400, detail="send {\"confirm\": true} to reset")
    get_ledger().reset()
    return {"reset": True}


# ---------------------------------------------------------------------- static
if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
