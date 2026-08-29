"""Funding providers.

**Read this before wiring up real money.**

quantlab never asks for, transmits or stores bank credentials, full account
numbers, routing numbers or card numbers. That is deliberate. Handling those
directly would put you in scope for PCI-DSS and banking regulation, and a
self-hosted SQLite file is not a safe place for them.

Instead, funding goes through a provider that owns the sensitive data:

* **Plaid** -- the user authenticates with their bank inside Plaid Link.
  Your server only ever sees an opaque ``access_token`` and the last four
  digits. https://plaid.com/docs/link/
* **Stripe** -- card and ACH details are collected by Stripe Elements in an
  iframe your JavaScript cannot read; you receive a ``PaymentMethod`` id.
  https://docs.stripe.com/payments/quickstart
* **Alpaca** -- a real brokerage with a free paper-trading environment; it
  handles funding and custody entirely. https://docs.alpaca.markets/

The default :class:`SandboxProvider` moves fictional money only, so you can
exercise the whole flow with nothing at risk. To go live, implement
:class:`FundingProvider` against one of the services above and pass it to
:class:`~quantlab.banking.ledger.Ledger`; nothing else in the codebase
changes.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

#: Fields quantlab refuses to persist, whatever the caller sends.
FORBIDDEN_FIELDS = frozenset(
    {
        "account_number", "full_account_number", "routing_number", "iban",
        "sort_code", "card_number", "pan", "cvv", "cvc", "pin", "password",
        "ssn", "tax_id", "secret", "access_token", "api_secret",
    }
)


class UnsafeFieldError(ValueError):
    """Raised when a caller tries to store sensitive banking data."""


def reject_sensitive_fields(payload: dict) -> None:
    """Raise if ``payload`` carries anything we refuse to store.

    This is a guard rail, not security theatre: it fires on the field name so
    that a well-meaning future change to the API or UI cannot start
    persisting card or account numbers by accident.
    """
    found = sorted(FORBIDDEN_FIELDS.intersection({k.lower() for k in payload}))
    if found:
        raise UnsafeFieldError(
            f"refusing to store sensitive field(s): {', '.join(found)}. "
            "quantlab stores only an institution name, an account type and the "
            "last four digits. Use a real provider (Plaid/Stripe/Alpaca) to "
            "handle full credentials -- see quantlab/banking/providers.py."
        )


def normalise_last4(value: str) -> str:
    """Reduce whatever the user typed to exactly four digits.

    Accepts a bare ``1234`` or a masked ``****1234``. A full account number is
    rejected outright rather than silently truncated -- truncating would mean
    the number reached the server, which is what we are avoiding.
    """
    raw = re.sub(r"[^0-9]", "", str(value or ""))
    if len(raw) > 4:
        raise UnsafeFieldError(
            "enter only the last four digits, not the full account number"
        )
    if len(raw) != 4:
        raise ValueError("last4 must be exactly four digits")
    return raw


@dataclass
class TransferResult:
    """What a provider returns when money is asked to move."""

    reference: str
    status: str  # settled | pending | failed
    message: str = ""
    settled_at: str | None = None


class FundingProvider(Protocol):
    """Interface a funding backend must implement."""

    name: str

    def link_account(self, institution: str, account_type: str, last4: str) -> str:
        """Register a funding source and return an opaque provider reference."""

    def deposit(self, reference: str, amount: float, currency: str) -> TransferResult:
        """Move ``amount`` from the funding source into the trading account."""

    def withdraw(self, reference: str, amount: float, currency: str) -> TransferResult:
        """Move ``amount`` from the trading account back to the source."""


class SandboxProvider:
    """The default provider: fictional money, settling instantly.

    Every transfer succeeds unless it breaks a limit, which keeps the demo
    predictable. Nothing leaves the machine.
    """

    name = "sandbox"

    #: Guard rails that mirror what a real ACH provider would enforce.
    min_transfer = 1.0
    max_transfer = 1_000_000.0

    def _ref(self, kind: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"sbx_{kind}_{stamp}_{secrets.token_hex(4)}"

    def link_account(self, institution: str, account_type: str, last4: str) -> str:
        return self._ref("acct")

    def _transfer(self, kind: str, amount: float) -> TransferResult:
        if not isinstance(amount, (int, float)) or amount != amount:
            return TransferResult("", "failed", "amount must be a number")
        if amount < self.min_transfer:
            return TransferResult("", "failed", f"minimum transfer is {self.min_transfer:.2f}")
        if amount > self.max_transfer:
            return TransferResult("", "failed", f"maximum transfer is {self.max_transfer:,.2f}")
        return TransferResult(
            reference=self._ref(kind),
            status="settled",
            message="sandbox transfer settled instantly (no real money moved)",
            settled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def deposit(self, reference: str, amount: float, currency: str = "USD") -> TransferResult:
        return self._transfer("dep", amount)

    def withdraw(self, reference: str, amount: float, currency: str = "USD") -> TransferResult:
        return self._transfer("wdl", amount)


class DelayedSandboxProvider(SandboxProvider):
    """Sandbox variant where deposits land as ``pending``.

    Useful for exercising the UI's pending/settled states, which is how a
    real ACH transfer behaves (T+1 to T+3).
    """

    name = "sandbox-delayed"

    def deposit(self, reference: str, amount: float, currency: str = "USD") -> TransferResult:
        result = self._transfer("dep", amount)
        if result.status == "settled":
            return TransferResult(
                result.reference, "pending", "ACH transfer initiated, settles in 1-3 business days"
            )
        return result


def get_provider(name: str = "sandbox") -> FundingProvider:
    """Look up a provider by name."""
    providers: dict[str, FundingProvider] = {
        "sandbox": SandboxProvider(),
        "sandbox-delayed": DelayedSandboxProvider(),
    }
    key = str(name or "sandbox").strip().lower()
    if key not in providers:
        raise KeyError(f"unknown funding provider {name!r}; available: {sorted(providers)}")
    return providers[key]
