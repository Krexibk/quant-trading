"""Ledger invariants: money is conserved and sensitive data is refused."""

import pytest

from quantlab.banking import InsufficientFunds, Ledger, LedgerError, UnsafeFieldError
from quantlab.banking.providers import DelayedSandboxProvider, normalise_last4


@pytest.fixture
def funded(ledger):
    ledger.link_account("Main", "Test Bank", "checking", "4321")
    ledger.deposit(100_000)
    return ledger


# ------------------------------------------------------------------ security
def test_rejects_full_account_number(ledger):
    with pytest.raises(UnsafeFieldError):
        ledger.link_account("Main", "Bank", "checking", "123456789012")


def test_rejects_sensitive_extra_fields(ledger):
    for field in ("routing_number", "card_number", "cvv", "password", "ssn"):
        with pytest.raises(UnsafeFieldError):
            ledger.link_account("M", "B", "checking", "4321", **{field: "secret"})


def test_stores_only_safe_metadata(ledger):
    acct = ledger.link_account("Main", "Chase", "checking", "4321")
    assert acct["last4"] == "4321"
    assert acct["masked"] == "****4321"
    assert "provider_ref" not in acct
    row = ledger._conn.execute("SELECT * FROM accounts").fetchone()
    stored = " ".join(str(v) for v in dict(row).values()).lower()
    for secret in ("123456789", "routing", "password", "cvv"):
        assert secret not in stored


@pytest.mark.parametrize("value,expected", [("1234", "1234"), ("****5678", "5678"), (" 0042 ", "0042")])
def test_normalise_last4(value, expected):
    assert normalise_last4(value) == expected


@pytest.mark.parametrize("bad", ["12345678", "12", "", "abc"])
def test_normalise_last4_rejects(bad):
    with pytest.raises(ValueError):
        normalise_last4(bad)


# ------------------------------------------------------------------- funding
def test_deposit_increases_cash(funded):
    assert funded.portfolio().cash == 100_000


def test_withdraw_decreases_cash(funded):
    funded.withdraw(25_000)
    assert funded.portfolio().cash == 75_000


def test_cannot_overdraw(funded):
    with pytest.raises(InsufficientFunds):
        funded.withdraw(200_000)
    assert funded.portfolio().cash == 100_000


@pytest.mark.parametrize("amount", [0, -100])
def test_rejects_non_positive_amounts(funded, amount):
    with pytest.raises(ValueError):
        funded.deposit(amount)


def test_transfer_without_account_fails(ledger):
    with pytest.raises(LedgerError):
        ledger.deposit(1000)


def test_pending_transfer_does_not_move_cash_until_settled(ledger):
    led = Ledger(":memory:", provider=DelayedSandboxProvider())
    led.link_account("M", "B", "checking", "4321")
    tr = led.deposit(5_000)
    assert tr["status"] == "pending"
    assert led.portfolio().cash == 0          # money has not arrived yet
    led.settle_transfer(tr["id"])
    assert led.portfolio().cash == 5_000
    with pytest.raises(LedgerError):
        led.settle_transfer(tr["id"])          # settling twice must not double-credit


# ------------------------------------------------------------------- trading
def test_buy_then_sell_round_trip(funded):
    buy = funded.place_order("AAPL", "buy", 100, 150.0)
    assert buy["status"] == "filled"
    p = funded.portfolio({"AAPL": 150.0})
    assert p.positions[0].quantity == 100
    assert p.cash == pytest.approx(100_000 - 15_000 - buy["fee"])

    sell = funded.place_order("AAPL", "sell", 100, 170.0)
    assert sell["status"] == "filled"
    p2 = funded.portfolio()
    assert not p2.positions
    assert p2.cash > 100_000  # a 20/share gain, net of two commissions


def test_average_cost_is_weighted(funded):
    funded.place_order("AAPL", "buy", 100, 100.0)
    funded.place_order("AAPL", "buy", 100, 200.0)
    assert funded.portfolio().positions[0].avg_price == pytest.approx(150.0)


def test_cannot_buy_beyond_cash(funded):
    order = funded.place_order("AAPL", "buy", 10_000, 150.0)
    assert order["status"] == "rejected"
    assert "insufficient funds" in order["reason"]
    assert funded.portfolio().cash == 100_000


def test_cannot_sell_more_than_held(funded):
    funded.place_order("AAPL", "buy", 10, 150.0)
    order = funded.place_order("AAPL", "sell", 50, 150.0)
    assert order["status"] == "rejected"
    assert funded.portfolio().positions[0].quantity == 10


def test_cannot_short_from_flat(funded):
    assert funded.place_order("AAPL", "sell", 10, 150.0)["status"] == "rejected"


def test_unmarketable_limit_stays_open(funded):
    order = funded.place_order("AAPL", "buy", 10, 150.0, "limit", limit_price=100.0)
    assert order["status"] == "open"
    assert funded.portfolio().cash == 100_000
    assert funded.cancel_order(order["id"])["status"] == "cancelled"


def test_marketable_limit_fills(funded):
    order = funded.place_order("AAPL", "buy", 10, 150.0, "limit", limit_price=160.0)
    assert order["status"] == "filled"
    assert order["filled_price"] == 150.0  # fills at the market, not the limit


@pytest.mark.parametrize("kwargs", [
    {"side": "hold"}, {"quantity": -5}, {"quantity": 0}, {"price": 0}, {"symbol": ""},
])
def test_order_validation(funded, kwargs):
    args = {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0, **kwargs}
    with pytest.raises(ValueError):
        funded.place_order(**args)


def test_realised_pnl_tracked(funded):
    funded.place_order("AAPL", "buy", 100, 100.0)
    funded.place_order("AAPL", "sell", 50, 120.0)
    assert funded.portfolio().positions[0].realised == pytest.approx(1000.0)


# ----------------------------------------------------------------- integrity
def test_integrity_holds_through_activity(funded):
    funded.place_order("AAPL", "buy", 50, 150.0)
    funded.place_order("MSFT", "buy", 20, 300.0)
    funded.place_order("AAPL", "sell", 25, 160.0)
    funded.withdraw(5_000)
    assert funded.verify_integrity()["ok"]


def test_pnl_measured_against_net_deposits(funded):
    funded.place_order("AAPL", "buy", 100, 100.0)
    p = funded.portfolio({"AAPL": 130.0})
    assert p.net_deposits == 100_000
    assert p.total_pnl == pytest.approx(3_000 - p.fees_paid, abs=0.01)


def test_unpriced_position_marks_at_cost(funded):
    funded.place_order("AAPL", "buy", 10, 100.0)
    pos = funded.portfolio().positions[0]   # no price supplied
    assert pos.last_price == 100.0
    assert pos.unrealised == 0.0


def test_reset_clears_everything(funded):
    funded.place_order("AAPL", "buy", 10, 150.0)
    funded.reset()
    p = funded.portfolio()
    assert p.cash == 0 and not p.positions and not funded.list_accounts()


def test_state_persists_across_connections(tmp_path):
    path = tmp_path / "ledger.db"
    led = Ledger(path)
    led.link_account("M", "B", "checking", "4321")
    led.deposit(10_000)
    led.place_order("AAPL", "buy", 10, 100.0)
    led.close()

    reopened = Ledger(path)
    assert reopened.portfolio().positions[0].quantity == 10
    assert reopened.verify_integrity()["ok"]
    reopened.close()


def test_integrity_survives_fractional_quantities(funded):
    """Cash is stored rounded to cents, so every amount that touches it must
    be rounded the same way. Unrounded notionals drift the balance away from
    the transaction log by fractions of a cent per trade.
    """
    for i, sym in enumerate(["AAPL", "MSFT", "NVDA", "GOOG", "TSLA", "AMZN"]):
        funded.place_order(sym, "buy", 13.7 + i * 0.37, 123.4567 + i * 7.77)
    funded.place_order("AAPL", "sell", 5.3, 131.1111)
    report = funded.verify_integrity()
    assert report["ok"], report
    assert report["difference"] == 0.0


def test_many_orders_do_not_accumulate_drift(funded):
    for i in range(60):
        funded.place_order("AAPL", "buy", 1.013, 99.9871 + i * 0.013)
    assert funded.verify_integrity()["ok"]
