"""Tests for the /v1/risk/scenario portfolio stress endpoint.

Success paths run the real engine; failure modes monkeypatch app.routers.risk.fdm_price_batch.
"""
import pytest

PATH = "/v1/risk/scenario"


@pytest.fixture
def btc_call():
    return {"deriv": 0, "s": 60000, "k": 65000, "time": 0.25, "sigma": 0.7,
            "r": 0.05, "q": 0.0, "Tn": 300, "h": 500.0}


@pytest.fixture
def matrix_request(btc_call):
    put = {**btc_call, "deriv": 1, "k": 55000}
    return {
        "portfolio": [
            {"config": btc_call, "quantity": 10},
            {"config": put, "quantity": -5},
        ],
        "shocks": [
            {"factor": "spot", "mode": "relative", "from": -0.2, "to": 0.2, "step": 0.1},
            {"factor": "sigma", "mode": "absolute", "from": -0.1, "to": 0.1, "step": 0.1},
        ],
    }


class TestScenarioSuccess:
    def test_matrix_shape_and_units(self, client, matrix_request):
        resp = client.post(PATH, json=matrix_request)
        assert resp.status_code == 200
        body = resp.json()
        assert body["positions"] == 2
        assert body["nodes"] == 15           # 5 spot × 3 vol
        assert body["pricing_units"] == 32   # 2 × (15 + 1 base block)
        assert len(body["grid"]) == 15
        assert body["axes"]["spot"] == [-0.2, -0.1, 0.0, 0.1, 0.2]

    def test_base_node_has_zero_pnl(self, client, matrix_request):
        # The unshocked node must price back to the base mark exactly.
        grid = client.post(PATH, json=matrix_request).json()["grid"]
        zero = next(n for n in grid if n["shocks"] == {"spot": 0.0, "sigma": 0.0})
        assert zero["pnl"] == pytest.approx(0.0, abs=1e-6)

    def test_summary_worst_case_matches_grid(self, client, matrix_request):
        body = client.post(PATH, json=matrix_request).json()
        min_pnl = min(n["pnl"] for n in body["grid"])
        assert body["summary"]["max_loss"] == pytest.approx(min_pnl)
        assert body["summary"]["worst_case"]["pnl"] == pytest.approx(min_pnl)

    def test_greeks_are_weighted_sum_not_average(self, client, btc_call):
        # Net delta of N identical long calls must be N× a single call's delta (sum, not mean).
        single = {"portfolio": [{"config": btc_call, "quantity": 1}],
                  "shocks": [{"factor": "spot", "values": [0.0]}]}
        ten = {"portfolio": [{"config": btc_call, "quantity": 10}],
               "shocks": [{"factor": "spot", "values": [0.0]}]}
        d1 = client.post(PATH, json=single).json()["grid"][0]["delta"]
        d10 = client.post(PATH, json=ten).json()["grid"][0]["delta"]
        assert d10 == pytest.approx(10 * d1, rel=1e-9)

    def test_one_dimensional_ladder(self, client, btc_call):
        req = {"portfolio": [{"config": btc_call, "quantity": 1}],
               "shocks": [{"factor": "spot", "mode": "relative", "from": -0.1, "to": 0.1, "step": 0.05}]}
        body = client.post(PATH, json=req).json()
        assert body["nodes"] == 5
        assert body["pricing_units"] == 6  # 1 × (5 + 1)


class TestScenarioValidation:
    def test_requires_auth(self, anon_client, matrix_request):
        assert anon_client.post(PATH, json=matrix_request).status_code == 401

    def test_empty_portfolio_is_422(self, client):
        req = {"portfolio": [], "shocks": [{"factor": "spot", "values": [0.0]}]}
        assert client.post(PATH, json=req).status_code == 422

    def test_duplicate_factor_is_422(self, client, btc_call):
        req = {"portfolio": [{"config": btc_call, "quantity": 1}],
               "shocks": [{"factor": "spot", "values": [0.0]}, {"factor": "spot", "values": [0.1]}]}
        assert client.post(PATH, json=req).status_code == 422

    def test_relative_spot_below_minus_one_is_422(self, client, btc_call):
        req = {"portfolio": [{"config": btc_call, "quantity": 1}],
               "shocks": [{"factor": "spot", "mode": "relative", "values": [-1.0]}]}
        assert client.post(PATH, json=req).status_code == 422

    def test_mixed_underlyings_under_spot_shock_is_422(self, client, btc_call):
        req = {"portfolio": [{"config": btc_call, "quantity": 1},
                             {"config": {**btc_call, "s": 3000}, "quantity": 1}],
               "shocks": [{"factor": "spot", "mode": "relative", "values": [0.1]}]}
        assert client.post(PATH, json=req).status_code == 422

    def test_oversize_scenario_is_413(self, client, btc_call, monkeypatch):
        monkeypatch.setattr("app.routers.risk.MAX_SCENARIO_VALUATIONS", 5)
        req = {"portfolio": [{"config": btc_call, "quantity": 1}],
               "shocks": [{"factor": "spot", "from": -0.2, "to": 0.2, "step": 0.05}]}  # 9 nodes
        assert client.post(PATH, json=req).status_code == 413


class TestScenarioEngineFailures:
    def test_license_expired_is_403(self, client, matrix_request, monkeypatch):
        monkeypatch.setattr("app.routers.risk.fdm_price_batch", lambda *a, **k: (-99, []))
        assert client.post(PATH, json=matrix_request).status_code == 403

    def test_engine_failure_is_500(self, client, matrix_request, monkeypatch):
        monkeypatch.setattr("app.routers.risk.fdm_price_batch", lambda *a, **k: (1, []))
        assert client.post(PATH, json=matrix_request).status_code == 500
