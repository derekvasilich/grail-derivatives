"""Shared fixtures for the FastAPI pricing endpoint test-suite.

Two authentication postures are provided:

* ``client``      — auth dependency overridden, simulating a valid bearer token.
* ``anon_client`` — no override, exercising the real JWT guard (expects 401).

Reusable binary-payload helpers live in ``tests/__init__.py`` alongside ``print_greeks``.
"""
import pytest
from fastapi.testclient import TestClient

from app.auth.jwt import UserClaims, get_current_user
from app.main import app


# ------------------------------------------------------------------
# Authentication postures
# ------------------------------------------------------------------
@pytest.fixture
def fake_user() -> UserClaims:
    """A stand-in for the claims FastAPI would extract from a verified JWT."""
    return UserClaims(sub="test-user-001", email="quant@example.com", username="quant_tester")


@pytest.fixture
def client(fake_user) -> TestClient:
    """TestClient with the auth dependency overridden — every request is 'authenticated'."""
    app.dependency_overrides[get_current_user] = lambda: fake_user
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def anon_client() -> TestClient:
    """TestClient with NO override, so the real bearer-token guard runs (expect 401)."""
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client


# ------------------------------------------------------------------
# Sample payloads
# ------------------------------------------------------------------
@pytest.fixture
def sample_config() -> dict:
    """A valid Vanilla Call OptionConfig JSON body (well-behaved, in-the-money-ish)."""
    return {
        "deriv": 0,  # VanillaCall
        "s": 100.0,
        "k": 110.0,
        "time": 1.0,
        "sigma": 0.5,
        "r": 0.1,
        "q": 0.0,
        "Tn": 300,
        "h": 1.0,
    }
