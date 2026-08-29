"""API contract tests."""

import pytest
from fastapi.testclient import TestClient

import quantlab.api as api_module
from quantlab.banking import Ledger


@pytest.fixture
def client():
    api_module.set_ledger(Ledger(":memory:"))
    api_module._cached_prices.cache_clear()
    with TestClient(api_module.app) as c:
        yield c


@pytest.fixture
def funded(client):
    client.post("/api/funding/accounts", json={
        "nickname": "Main", "institution": "Test Bank",
        "account_type": "checking", "last4": "4321",
    })
    client.post("/api/funding/deposit", json={"amount": 100_000})
    return client


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["paper_trading"] is True


def test_strategies_listed(client):
    body = client.get("/api/strategies").json()
    assert len(body["strategies"]) >= 9
    assert all(s["name"] and s["params"] is not None for s in body["strategies"])


def test_quote_and_history(client):
    q = client.get("/api/quote/AAPL?source=synthetic").json()
    assert q["symbol"] == "AAPL" and q["price"] > 0
    h = client.get("/api/history/AAPL?days=90&source=synthetic").json()
    assert len(h["dates"]) == len(h["close"]) == 90


def test_link_account_masks_data(client):
    r = client.post("/api/funding/accounts", json={
        "nickname": "Main", "institution": "Chase",
        "account_type": "checking", "last4": "4321",
    })
    assert r.status_code == 201
    assert r.json()["masked"] == "****4321"


def test_api_rejects_full_account_number(client):
    r = client.post("/api/funding/accounts", json={
        "nickname": "M", "institution": "B",
        "account_type": "checking", "last4": "123456789012",
    })
    assert r.status_code == 422


@pytest.mark.parametrize("field", ["routing_number", "card_number", "cvv", "password"])
def test_api_refuses_sensitive_fields(client, field):
    """Unknown fields are refused loudly rather than silently dropped."""
    r = client.post("/api/funding/accounts", json={
        "nickname": "M", "institution": "B", "account_type": "checking",
        "last4": "4321", field: "1234567",
    })
    assert r.status_code == 422


def test_deposit_and_portfolio(funded):
    assert funded.get("/api/portfolio").json()["cash"] == 100_000


def test_overdraw_returns_402(funded):
    r = funded.post("/api/funding/withdraw", json={"amount": 500_000})
    assert r.status_code == 402


def test_order_flow(funded):
    r = funded.post("/api/orders?source=synthetic",
                    json={"symbol": "AAPL", "side": "buy", "quantity": 5})
    assert r.status_code == 201 and r.json()["status"] == "filled"
    p = funded.get("/api/portfolio?source=synthetic").json()
    assert p["positions"][0]["symbol"] == "AAPL"
    assert funded.get("/api/integrity").json()["ok"]


def test_client_cannot_dictate_fill_price(funded):
    """A price sent by the client must be ignored.

    If it were honoured, anyone could buy at $0.01 and mint money.
    """
    r = funded.post("/api/orders?source=synthetic", json={
        "symbol": "AAPL", "side": "buy", "quantity": 1, "price": 0.01,
    })
    assert r.status_code in (201, 422)
    if r.status_code == 201:
        assert r.json()["filled_price"] > 1.0


def test_oversell_rejected(funded):
    r = funded.post("/api/orders?source=synthetic",
                    json={"symbol": "AAPL", "side": "sell", "quantity": 100})
    assert r.json()["status"] == "rejected"


@pytest.mark.parametrize("payload", [
    {"symbol": "AAPL", "side": "sideways", "quantity": 1},
    {"symbol": "AAPL", "side": "buy", "quantity": -1},
    {"symbol": "", "side": "buy", "quantity": 1},
])
def test_order_validation(funded, payload):
    assert funded.post("/api/orders?source=synthetic", json=payload).status_code == 422


def test_backtest(client):
    r = client.post("/api/backtest", json={
        "strategy": "macd", "symbol": "AAPL", "source": "synthetic",
        "start": "2020-01-01", "end": "2024-12-31",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["equity"]) == len(body["dates"]) == len(body["drawdown"])
    assert "sharpe" in body["stats"]


def test_backtest_unknown_strategy(client):
    assert client.post("/api/backtest", json={"strategy": "nope"}).status_code == 404


def test_pairs_requires_second_symbol(client):
    r = client.post("/api/backtest", json={
        "strategy": "pairs", "symbol": "AAPL", "source": "synthetic",
    })
    assert r.status_code == 400


def test_pairs_with_second_symbol(client):
    r = client.post("/api/backtest", json={
        "strategy": "pairs", "symbol": "KO", "symbol_b": "PEP",
        "source": "synthetic", "start": "2020-01-01",
    })
    assert r.status_code == 200


def test_reset_requires_confirmation(funded):
    assert funded.post("/api/admin/reset", json={"confirm": False}).status_code == 400
    assert funded.post("/api/admin/reset", json={"confirm": True}).status_code == 200
    assert funded.get("/api/portfolio").json()["cash"] == 0


def test_ui_is_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
