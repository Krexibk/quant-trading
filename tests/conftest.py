import os

# Keep the whole suite deterministic and off the network.
os.environ["QUANTLAB_ALLOW_NETWORK"] = "0"

import pytest

from quantlab.data import synthetic_prices


@pytest.fixture(scope="session")
def prices():
    return synthetic_prices("TEST", periods=1200)


@pytest.fixture(scope="session")
def prices_b():
    return synthetic_prices("TEST2", periods=1200)


@pytest.fixture
def ledger():
    from quantlab.banking import Ledger

    led = Ledger(":memory:")
    yield led
    led.close()
