"""A double-entry paper-trading ledger backed by SQLite.

Money and positions are tracked with integer-safe rounding and every mutation
happens inside a transaction, so a crash mid-order cannot leave the account
holding a position it never paid for.

Cash is the single source of truth: ``cash`` moves only through recorded
transfers and fills, and :meth:`Ledger.verify_integrity` re-derives it from
the transaction log to prove nothing has drifted.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantlab import config
from quantlab.banking.providers import (
    FundingProvider,
    get_provider,
    normalise_last4,
    reject_sensitive_fields,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id            TEXT PRIMARY KEY,
    nickname      TEXT NOT NULL,
    institution   TEXT NOT NULL,
    account_type  TEXT NOT NULL,
    last4         TEXT NOT NULL,
    provider      TEXT NOT NULL,
    provider_ref  TEXT NOT NULL,
    currency      TEXT NOT NULL DEFAULT 'USD',
    created_at    TEXT NOT NULL,
    is_default    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transfers (
    id          TEXT PRIMARY KEY,
    account_id  TEXT REFERENCES accounts(id),
    kind        TEXT NOT NULL CHECK (kind IN ('deposit','withdrawal')),
    amount      REAL NOT NULL CHECK (amount > 0),
    currency    TEXT NOT NULL DEFAULT 'USD',
    status      TEXT NOT NULL CHECK (status IN ('pending','settled','failed','cancelled')),
    reference   TEXT,
    message     TEXT,
    created_at  TEXT NOT NULL,
    settled_at  TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id           TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL CHECK (side IN ('buy','sell')),
    order_type   TEXT NOT NULL CHECK (order_type IN ('market','limit')),
    quantity     REAL NOT NULL CHECK (quantity > 0),
    limit_price  REAL,
    status       TEXT NOT NULL CHECK (status IN ('filled','rejected','cancelled','open')),
    filled_price REAL,
    fee          REAL NOT NULL DEFAULT 0,
    reason       TEXT,
    strategy     TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol     TEXT PRIMARY KEY,
    quantity   REAL NOT NULL DEFAULT 0,
    avg_price  REAL NOT NULL DEFAULT 0,
    realised   REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS idx_transfers_created ON transfers(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_created    ON orders(created_at DESC);
"""

#: Commission charged per trade, as a fraction of notional.
DEFAULT_FEE_RATE = 0.0005
CASH_KEY = "cash"


class LedgerError(RuntimeError):
    """Raised when an operation would violate an account invariant."""


class InsufficientFunds(LedgerError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _money(value: float) -> float:
    """Round to cents. Floats accumulate error; money must not."""
    return round(float(value) + 0.0, 2)


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    realised: float
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        return _money(self.quantity * self.last_price)

    @property
    def cost_basis(self) -> float:
        return _money(self.quantity * self.avg_price)

    @property
    def unrealised(self) -> float:
        return _money(self.market_value - self.cost_basis)

    @property
    def unrealised_pct(self) -> float:
        basis = abs(self.cost_basis)
        return (self.unrealised / basis) if basis > 1e-9 else 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "quantity": round(self.quantity, 6),
            "avg_price": round(self.avg_price, 4), "last_price": round(self.last_price, 4),
            "market_value": self.market_value, "cost_basis": self.cost_basis,
            "unrealised": self.unrealised, "unrealised_pct": round(self.unrealised_pct, 6),
            "realised": _money(self.realised),
        }


@dataclass
class Portfolio:
    cash: float
    positions: list[Position] = field(default_factory=list)
    deposited: float = 0.0
    withdrawn: float = 0.0
    fees_paid: float = 0.0

    @property
    def positions_value(self) -> float:
        return _money(sum(p.market_value for p in self.positions))

    @property
    def equity(self) -> float:
        return _money(self.cash + self.positions_value)

    @property
    def net_deposits(self) -> float:
        return _money(self.deposited - self.withdrawn)

    @property
    def total_pnl(self) -> float:
        """Profit measured against money actually put in, not a fixed start."""
        return _money(self.equity - self.net_deposits)

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl / self.net_deposits) if self.net_deposits > 1e-9 else 0.0

    def to_dict(self) -> dict:
        return {
            "cash": _money(self.cash), "positions_value": self.positions_value,
            "equity": self.equity, "deposited": _money(self.deposited),
            "withdrawn": _money(self.withdrawn), "net_deposits": self.net_deposits,
            "fees_paid": _money(self.fees_paid), "total_pnl": self.total_pnl,
            "total_pnl_pct": round(self.total_pnl_pct, 6),
            "positions": [p.to_dict() for p in self.positions],
        }


class Ledger:
    """Paper-trading account persisted in SQLite."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        provider: FundingProvider | str = "sandbox",
        fee_rate: float = DEFAULT_FEE_RATE,
    ) -> None:
        config.ensure_dirs()
        self.db_path = Path(db_path) if db_path else config.DB_PATH
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.provider = get_provider(provider) if isinstance(provider, str) else provider
        self.fee_rate = float(fee_rate)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._tx() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (CASH_KEY, "0.0")
            )

    # ----------------------------------------------------------------- plumbing
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Run a block atomically; roll back on any exception."""
        try:
            with self._conn:
                yield self._conn
        except sqlite3.IntegrityError as exc:
            raise LedgerError(str(exc)) from exc

    def close(self) -> None:
        self._conn.close()

    def _get_cash(self, conn: sqlite3.Connection | None = None) -> float:
        conn = conn or self._conn
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (CASH_KEY,)).fetchone()
        return float(row["value"]) if row else 0.0

    def _set_cash(self, conn: sqlite3.Connection, value: float) -> None:
        if value < -1e-6:
            raise InsufficientFunds("operation would overdraw the account")
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = ?", (str(_money(max(value, 0.0))), CASH_KEY)
        )

    # ------------------------------------------------------------------ funding
    def link_account(
        self,
        nickname: str,
        institution: str,
        account_type: str,
        last4: str,
        currency: str = "USD",
        make_default: bool = True,
        **extra: Any,
    ) -> dict:
        """Register a funding source.

        Only non-sensitive metadata is stored. Passing a full account number,
        routing number or card number raises :class:`UnsafeFieldError`.
        """
        reject_sensitive_fields(extra)
        nickname = str(nickname or "").strip()
        institution = str(institution or "").strip()
        account_type = str(account_type or "checking").strip().lower()
        if not nickname:
            raise ValueError("nickname is required")
        if not institution:
            raise ValueError("institution is required")
        if account_type not in {"checking", "savings", "brokerage", "card"}:
            raise ValueError("account_type must be checking, savings, brokerage or card")
        digits = normalise_last4(last4)

        ref = self.provider.link_account(institution, account_type, digits)
        account_id = _uid("acct")
        with self._tx() as conn:
            if make_default:
                conn.execute("UPDATE accounts SET is_default = 0")
            conn.execute(
                "INSERT INTO accounts (id, nickname, institution, account_type, last4,"
                " provider, provider_ref, currency, created_at, is_default)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (account_id, nickname, institution, account_type, digits,
                 self.provider.name, ref, currency.upper(), _now(), int(make_default)),
            )
        return self.get_account(account_id)

    def get_account(self, account_id: str) -> dict:
        row = self._conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        if row is None:
            raise LedgerError(f"no funding account {account_id!r}")
        return self._account_dict(row)

    @staticmethod
    def _account_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["is_default"] = bool(d["is_default"])
        d["masked"] = f"****{d['last4']}"
        d.pop("provider_ref", None)  # opaque, never needed by a client
        return d

    def list_accounts(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM accounts ORDER BY is_default DESC, created_at DESC"
        ).fetchall()
        return [self._account_dict(r) for r in rows]

    def remove_account(self, account_id: str) -> None:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
            if cur.rowcount == 0:
                raise LedgerError(f"no funding account {account_id!r}")

    def deposit(self, amount: float, account_id: str | None = None) -> dict:
        """Add funds from a linked source."""
        return self._transfer("deposit", amount, account_id)

    def withdraw(self, amount: float, account_id: str | None = None) -> dict:
        """Send funds back to a linked source."""
        return self._transfer("withdrawal", amount, account_id)

    def _default_account_id(self) -> str:
        row = self._conn.execute(
            "SELECT id FROM accounts ORDER BY is_default DESC, created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise LedgerError("link a funding account before transferring money")
        return row["id"]

    def _transfer(self, kind: str, amount: float, account_id: str | None) -> dict:
        try:
            amount = _money(amount)
        except (TypeError, ValueError):
            raise ValueError("amount must be a number") from None
        if amount <= 0:
            raise ValueError("amount must be positive")

        account_id = account_id or self._default_account_id()
        row = self._conn.execute(
            "SELECT * FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"no funding account {account_id!r}")

        if kind == "withdrawal" and amount > self._get_cash() + 1e-9:
            raise InsufficientFunds(
                f"cannot withdraw {amount:,.2f}: settled cash is {self._get_cash():,.2f}"
            )

        call = self.provider.deposit if kind == "deposit" else self.provider.withdraw
        result = call(row["provider_ref"], amount, row["currency"])

        transfer_id = _uid("tr")
        with self._tx() as conn:
            # Cash moves only when the provider says the money actually settled.
            if result.status == "settled":
                cash = self._get_cash(conn)
                self._set_cash(conn, cash + amount if kind == "deposit" else cash - amount)
            conn.execute(
                "INSERT INTO transfers (id, account_id, kind, amount, currency, status,"
                " reference, message, created_at, settled_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (transfer_id, account_id, kind, amount, row["currency"], result.status,
                 result.reference, result.message, _now(), result.settled_at),
            )
        if result.status == "failed":
            raise LedgerError(result.message or "transfer rejected by provider")
        return self.get_transfer(transfer_id)

    def settle_transfer(self, transfer_id: str) -> dict:
        """Mark a pending transfer settled and move the cash.

        A real deployment calls this from the provider's webhook.
        """
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM transfers WHERE id = ?", (transfer_id,)).fetchone()
            if row is None:
                raise LedgerError(f"no transfer {transfer_id!r}")
            if row["status"] != "pending":
                raise LedgerError(f"transfer is {row['status']}, not pending")
            cash = self._get_cash(conn)
            amount = float(row["amount"])
            if row["kind"] == "withdrawal" and amount > cash + 1e-9:
                raise InsufficientFunds("insufficient cash to settle withdrawal")
            self._set_cash(conn, cash + amount if row["kind"] == "deposit" else cash - amount)
            conn.execute(
                "UPDATE transfers SET status='settled', settled_at=? WHERE id=?",
                (_now(), transfer_id),
            )
        return self.get_transfer(transfer_id)

    def get_transfer(self, transfer_id: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM transfers WHERE id = ?", (transfer_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"no transfer {transfer_id!r}")
        return dict(row)

    def list_transfers(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT t.*, a.nickname, a.last4 FROM transfers t"
            " LEFT JOIN accounts a ON a.id = t.account_id"
            " ORDER BY t.created_at DESC, t.rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ trading
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "market",
        limit_price: float | None = None,
        strategy: str | None = None,
    ) -> dict:
        """Place and immediately attempt to fill an order.

        ``price`` is the current market price, supplied by the caller so the
        ledger never has to reach the network. A limit order fills only if
        the market price is at or better than the limit.
        """
        symbol = str(symbol or "").strip().upper()
        side = str(side or "").strip().lower()
        order_type = str(order_type or "market").strip().lower()
        if not symbol:
            raise ValueError("symbol is required")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if order_type not in {"market", "limit"}:
            raise ValueError("order_type must be 'market' or 'limit'")
        quantity = float(quantity)
        if quantity <= 0 or quantity != quantity:
            raise ValueError("quantity must be positive")
        price = float(price)
        if price <= 0:
            raise ValueError("price must be positive")

        order_id = _uid("ord")
        fill_price = price
        if order_type == "limit":
            if limit_price is None or float(limit_price) <= 0:
                raise ValueError("limit orders need a positive limit_price")
            limit_price = float(limit_price)
            marketable = price <= limit_price if side == "buy" else price >= limit_price
            if not marketable:
                return self._record_order(
                    order_id, symbol, side, order_type, quantity, limit_price,
                    "open", None, 0.0,
                    f"limit {limit_price:,.2f} not marketable at {price:,.2f}", strategy,
                )
            # Price improvement is not free money: fill at the market.
            fill_price = price

        # Round the notional to cents *before* it touches cash. Cash is stored
        # rounded, so if the amount debited were unrounded the balance and the
        # transaction log would drift apart by fractions of a cent per trade --
        # enough to fail an integrity check after a handful of orders.
        notional = _money(quantity * fill_price)
        fee = _money(notional * self.fee_rate)

        with self._tx() as conn:
            cash = self._get_cash(conn)
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            held = float(pos["quantity"]) if pos else 0.0
            avg = float(pos["avg_price"]) if pos else 0.0
            realised = float(pos["realised"]) if pos else 0.0

            if side == "buy":
                if notional + fee > cash + 1e-9:
                    return self._record_order(
                        order_id, symbol, side, order_type, quantity, limit_price,
                        "rejected", None, 0.0,
                        f"insufficient funds: need {notional + fee:,.2f}, have {cash:,.2f}",
                        strategy, conn=conn,
                    )
                new_qty = held + quantity
                # Weighted average cost; this is what unrealised P&L is measured from.
                avg = (held * avg + quantity * fill_price) / new_qty if new_qty else 0.0
                self._set_cash(conn, cash - notional - fee)
            else:
                # No short selling in the paper account: you cannot sell what
                # you do not hold. Shorting needs margin and a borrow, which
                # this ledger does not model.
                if quantity > held + 1e-9:
                    return self._record_order(
                        order_id, symbol, side, order_type, quantity, limit_price,
                        "rejected", None, 0.0,
                        f"insufficient position: hold {held:g}, tried to sell {quantity:g}",
                        strategy, conn=conn,
                    )
                realised += (fill_price - avg) * quantity
                new_qty = held - quantity
                if new_qty <= 1e-9:
                    new_qty, avg = 0.0, 0.0
                self._set_cash(conn, cash + notional - fee)

            conn.execute(
                "INSERT INTO positions (symbol, quantity, avg_price, realised, updated_at)"
                " VALUES (?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET"
                " quantity=excluded.quantity, avg_price=excluded.avg_price,"
                " realised=excluded.realised, updated_at=excluded.updated_at",
                (symbol, new_qty, avg, realised, _now()),
            )
            return self._record_order(
                order_id, symbol, side, order_type, quantity, limit_price,
                "filled", fill_price, fee, None, strategy, conn=conn,
            )

    def _record_order(
        self, order_id, symbol, side, order_type, quantity, limit_price,
        status, filled_price, fee, reason, strategy, conn=None,
    ) -> dict:
        sql = (
            "INSERT INTO orders (id, symbol, side, order_type, quantity, limit_price,"
            " status, filled_price, fee, reason, strategy, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        args = (order_id, symbol, side, order_type, quantity, limit_price, status,
                filled_price, fee, reason, strategy, _now())
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self._tx() as c:
                c.execute(sql, args)
        return {
            "id": order_id, "symbol": symbol, "side": side, "order_type": order_type,
            "quantity": quantity, "limit_price": limit_price, "status": status,
            "filled_price": filled_price, "fee": fee, "reason": reason,
            "strategy": strategy, "created_at": _now(),
        }

    def list_orders(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM orders ORDER BY created_at DESC, rowid DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [dict(r) for r in rows]

    def cancel_order(self, order_id: str) -> dict:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
            if row is None:
                raise LedgerError(f"no order {order_id!r}")
            if row["status"] != "open":
                raise LedgerError(f"order is {row['status']}, only open orders can be cancelled")
            conn.execute("UPDATE orders SET status='cancelled' WHERE id=?", (order_id,))
        return dict(self._conn.execute(
            "SELECT * FROM orders WHERE id = ?", (order_id,)
        ).fetchone())

    # ---------------------------------------------------------------- portfolio
    def portfolio(self, prices: dict[str, float] | None = None) -> Portfolio:
        """Current cash, positions and P&L.

        ``prices`` maps symbol -> last price. Symbols missing from it are
        marked at their average cost, so unrealised P&L reads zero rather
        than a misleading number.
        """
        prices = {k.upper(): float(v) for k, v in (prices or {}).items()}
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE quantity > 1e-9 ORDER BY symbol"
        ).fetchall()
        positions = [
            Position(
                symbol=r["symbol"], quantity=float(r["quantity"]),
                avg_price=float(r["avg_price"]), realised=float(r["realised"]),
                last_price=prices.get(r["symbol"], float(r["avg_price"])),
            )
            for r in rows
        ]
        agg = self._conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN kind='deposit'    AND status='settled'"
            "                          THEN amount ELSE 0 END), 0) AS dep,"
            "       COALESCE(SUM(CASE WHEN kind='withdrawal' AND status='settled'"
            "                          THEN amount ELSE 0 END), 0) AS wdl FROM transfers"
        ).fetchone()
        fees = self._conn.execute(
            "SELECT COALESCE(SUM(fee), 0) AS f FROM orders WHERE status='filled'"
        ).fetchone()["f"]
        return Portfolio(
            cash=self._get_cash(), positions=positions,
            deposited=float(agg["dep"]), withdrawn=float(agg["wdl"]), fees_paid=float(fees),
        )

    def verify_integrity(self) -> dict:
        """Re-derive cash from the transaction log and compare it to the balance.

        Any mismatch means a bug or external tampering, so this is worth
        running in tests and after a crash.
        """
        agg = self._conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN kind='deposit' AND status='settled'"
            "                          THEN amount ELSE -0.0 END),0)"
            "     - COALESCE(SUM(CASE WHEN kind='withdrawal' AND status='settled'"
            "                          THEN amount ELSE 0 END),0) AS net FROM transfers"
        ).fetchone()
        # ROUND per row, mirroring the per-order rounding above; summing the
        # raw products and rounding once at the end would not reconcile.
        trades = self._conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN side='sell' THEN ROUND(quantity*filled_price, 2)"
            "                         ELSE -ROUND(quantity*filled_price, 2) END),0) AS flow,"
            "       COALESCE(SUM(fee),0) AS fees FROM orders WHERE status='filled'"
        ).fetchone()
        expected = _money(float(agg["net"]) + float(trades["flow"]) - float(trades["fees"]))
        actual = _money(self._get_cash())
        return {
            "expected_cash": expected,
            "actual_cash": actual,
            "difference": _money(actual - expected),
            "ok": abs(actual - expected) < 0.01,
        }

    def reset(self) -> None:
        """Wipe all state. Used by tests and the UI's 'reset demo' button."""
        with self._tx() as conn:
            for table in ("orders", "positions", "transfers", "accounts"):
                conn.execute(f"DELETE FROM {table}")
            conn.execute("UPDATE meta SET value='0.0' WHERE key=?", (CASH_KEY,))
