"""Comprehensive HTTP-layer tests for the /v1/pricing/* API routes.

Coverage per endpoint:
  * success path through the real compiled C++ engine,
  * authentication guard (401 when no bearer token),
  * request validation (422 on bad/missing fields),
  * endpoint-specific 400s (empty batch, misaligned binary),
  * engine failure modes faked via monkeypatch (403 license expiry, 500 solver crash).

Success-path tests exercise the real `.so`/`.dylib`; failure-mode tests monkeypatch the
engine entry points (`app.routers.pricing.fdm_price_*`) so they run fast and deterministically.
"""
import ctypes

import numpy as np
import pytest

from app.api import OptionGreeks
from tests import (
    CONFIG_STRUCT_SIZE,
    GREEKS_STRUCT_SIZE,
    build_config_struct,
    pack_configs,
    unpack_greeks,
)

PREFIX = "/v1/pricing"

# Endpoints that take a single JSON OptionConfig body, used for cross-cutting auth checks.
JSON_BODY_ENDPOINTS = [f"{PREFIX}/single", f"{PREFIX}/grid", f"{PREFIX}/chart"]


# ------------------------------------------------------------------
# Engine fakes for failure-mode injection
# ------------------------------------------------------------------
def _fake_greeks(tn: int = 300, xm: int = 721) -> OptionGreeks:
    g = OptionGreeks()
    g.price, g.delta, g.gamma, g.theta, g.vega = 1.0, 0.5, 0.01, -0.1, 0.0
    g.Tn, g.Xm = tn, xm
    return g


def _patch_single(monkeypatch, status, ptr_value=12345):
    """Make fdm_price_single return a given status (and optional surface pointer)."""
    ptr = ctypes.c_void_p(ptr_value)
    monkeypatch.setattr(
        "app.routers.pricing.fdm_price_single",
        lambda *a, **k: (status, _fake_greeks(), ptr),
    )


def _patch_batch(monkeypatch, status):
    monkeypatch.setattr(
        "app.routers.pricing.fdm_price_batch",
        lambda *a, **k: (status, []),
    )


# ==================================================================
# POST /pricing/single
# ==================================================================
class TestSingle:
    def test_success_returns_greeks(self, client, sample_config):
        resp = client.post(f"{PREFIX}/single", json=sample_config)
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"price", "delta", "gamma", "theta", "vega", "Tn", "Xm"}
        # Vanilla call: price positive, delta in [0,1], gamma positive, theta negative.
        assert body["price"] > 0
        assert 0.0 <= body["delta"] <= 1.0
        assert body["gamma"] > 0
        assert body["theta"] < 0
        assert body["Tn"] == sample_config["Tn"]
        assert body["Xm"] > 0

    @pytest.mark.parametrize("vega_flag", [True, False])
    def test_calculate_vega_flag_accepted(self, client, sample_config, vega_flag):
        resp = client.post(
            f"{PREFIX}/single", params={"calculate_vega": vega_flag}, json=sample_config
        )
        assert resp.status_code == 200
        # TODO(vega): only asserts the field type because the shared lib does not yet
        # populate vega (always 0.0). See test_vega_populated_when_requested below — once
        # vega is implemented, tighten the calculate_vega=True case to assert vega != 0.0.
        assert isinstance(resp.json()["vega"], float)

    @pytest.mark.xfail(
        reason="TODO(vega): the shared lib does not yet compute vega — it returns 0.0 even "
               "with calculate_vega=true. Remove this xfail once vega is implemented in libblackfdmcore.",
        strict=False,
    )
    def test_vega_populated_when_requested(self, client, sample_config):
        # When calculate_vega=true, an in-the-money-ish call should carry a meaningful vega.
        # Magnitude threshold (rather than != 0.0) so the test only passes on a genuine
        # computed sensitivity, not on a near-zero rounding artifact.
        resp = client.post(f"{PREFIX}/single", params={"calculate_vega": True}, json=sample_config)
        assert resp.status_code == 200
        assert abs(resp.json()["vega"]) > 1e-6

    def test_requires_auth(self, anon_client, sample_config):
        assert anon_client.post(f"{PREFIX}/single", json=sample_config).status_code == 401

    def test_missing_required_deriv_is_422(self, client, sample_config):
        sample_config.pop("deriv")
        assert client.post(f"{PREFIX}/single", json=sample_config).status_code == 422

    def test_deriv_out_of_range_is_422(self, client, sample_config):
        sample_config["deriv"] = 99  # schema constrains 0..5
        assert client.post(f"{PREFIX}/single", json=sample_config).status_code == 422

    def test_non_numeric_field_is_422(self, client, sample_config):
        sample_config["sigma"] = "not-a-number"
        assert client.post(f"{PREFIX}/single", json=sample_config).status_code == 422

    def test_license_expired_is_403(self, client, sample_config, monkeypatch):
        _patch_single(monkeypatch, -99)
        resp = client.post(f"{PREFIX}/single", json=sample_config)
        assert resp.status_code == 403
        assert "license" in resp.json()["detail"].lower()

    def test_engine_failure_is_500(self, client, sample_config, monkeypatch):
        _patch_single(monkeypatch, 1)
        assert client.post(f"{PREFIX}/single", json=sample_config).status_code == 500


# ==================================================================
# POST /pricing/batch
# ==================================================================
class TestBatch:
    def test_success_single_item(self, client, sample_config):
        resp = client.post(f"{PREFIX}/batch", json=[sample_config])
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list) and len(body) == 1
        assert body[0]["price"] > 0

    def test_index_alignment_preserved(self, client, sample_config):
        cheap = {**sample_config, "k": 130.0}  # deeper OTM call -> lower premium
        rich = {**sample_config, "k": 90.0}    # ITM call -> higher premium
        resp = client.post(f"{PREFIX}/batch", json=[cheap, rich])
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        # Element i of the response must correspond to element i of the request.
        assert body[0]["price"] < body[1]["price"]

    @pytest.mark.xfail(
        reason="TODO(vega): the shared lib does not yet compute vega — batch results return 0.0 "
               "even with calculate_vega=true. Remove this xfail once vega is implemented in libblackfdmcore.",
        strict=False,
    )
    def test_vega_populated_when_requested(self, client, sample_config):
        # Every batch element should carry a meaningful vega when calculate_vega=true.
        # Magnitude threshold (rather than != 0.0) so the test only passes on a genuine
        # computed sensitivity, not on a near-zero rounding artifact.
        resp = client.post(f"{PREFIX}/batch", params={"calculate_vega": True}, json=[sample_config])
        assert resp.status_code == 200
        assert all(abs(item["vega"]) > 1e-6 for item in resp.json())

    def test_empty_batch_is_400(self, client):
        resp = client.post(f"{PREFIX}/batch", json=[])
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_requires_auth(self, anon_client, sample_config):
        assert anon_client.post(f"{PREFIX}/batch", json=[sample_config]).status_code == 401

    def test_non_list_body_is_422(self, client, sample_config):
        assert client.post(f"{PREFIX}/batch", json=sample_config).status_code == 422

    def test_invalid_item_in_batch_is_422(self, client, sample_config):
        bad = {**sample_config, "deriv": 99}
        assert client.post(f"{PREFIX}/batch", json=[sample_config, bad]).status_code == 422

    def test_license_expired_is_403(self, client, sample_config, monkeypatch):
        _patch_batch(monkeypatch, -99)
        resp = client.post(f"{PREFIX}/batch", json=[sample_config])
        assert resp.status_code == 403

    def test_engine_failure_is_500(self, client, sample_config, monkeypatch):
        _patch_batch(monkeypatch, 1)
        assert client.post(f"{PREFIX}/batch", json=[sample_config]).status_code == 500


# ==================================================================
# POST /pricing/binary
# ==================================================================
class TestBinary:
    def test_success_roundtrip(self, client):
        configs = [build_config_struct(k=k) for k in (90.0, 110.0, 130.0)]
        payload = pack_configs(configs)
        resp = client.post(
            f"{PREFIX}/binary",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        # One Greeks struct per input config, byte-exact.
        assert len(resp.content) == len(configs) * GREEKS_STRUCT_SIZE
        greeks = unpack_greeks(resp.content)
        assert all(g.price > 0 for g in greeks)

    def test_empty_body_is_400(self, client):
        resp = client.post(
            f"{PREFIX}/binary", content=b"", headers={"Content-Type": "application/octet-stream"}
        )
        assert resp.status_code == 400

    def test_misaligned_body_is_400(self, client):
        # One byte over a single struct -> not a clean multiple of the struct size.
        payload = pack_configs([build_config_struct()]) + b"\x00"
        assert payload.__len__() % CONFIG_STRUCT_SIZE != 0
        resp = client.post(
            f"{PREFIX}/binary",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 400
        assert "multiple" in resp.json()["detail"].lower()

    def test_requires_auth(self, anon_client):
        payload = pack_configs([build_config_struct()])
        resp = anon_client.post(
            f"{PREFIX}/binary",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 401

    def test_license_expired_is_403(self, client, monkeypatch):
        _patch_batch(monkeypatch, -99)
        payload = pack_configs([build_config_struct()])
        resp = client.post(
            f"{PREFIX}/binary",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 403

    def test_engine_failure_is_500(self, client, monkeypatch):
        _patch_batch(monkeypatch, 1)
        payload = pack_configs([build_config_struct()])
        resp = client.post(
            f"{PREFIX}/binary",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 500


# ==================================================================
# POST /pricing/grid
# ==================================================================
class TestGrid:
    def test_success_surface_and_headers(self, client, sample_config):
        resp = client.post(f"{PREFIX}/grid", json=sample_config)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"

        # All scalar Greeks + geometry must ride in the X-* headers.
        for header in ("X-Price", "X-Delta", "X-Gamma", "X-Theta", "X-Vega",
                       "X-Grid-Rows-Tn", "X-Grid-Cols-Xm"):
            assert header in resp.headers

        rows = int(resp.headers["X-Grid-Rows-Tn"])
        cols = int(resp.headers["X-Grid-Cols-Xm"])
        # Body is a flat row-major float64 array of exactly Tn * Xm elements.
        assert len(resp.content) == rows * cols * 8
        surface = np.frombuffer(resp.content, dtype="<f8").reshape((rows, cols))
        assert surface.shape == (rows, cols)
        assert np.isfinite(surface).all()

    def test_requires_auth(self, anon_client, sample_config):
        assert anon_client.post(f"{PREFIX}/grid", json=sample_config).status_code == 401

    def test_invalid_body_is_422(self, client, sample_config):
        sample_config["deriv"] = 99
        assert client.post(f"{PREFIX}/grid", json=sample_config).status_code == 422

    def test_license_expired_is_403(self, client, sample_config, monkeypatch):
        _patch_single(monkeypatch, -99)
        assert client.post(f"{PREFIX}/grid", json=sample_config).status_code == 403

    def test_engine_failure_is_500(self, client, sample_config, monkeypatch):
        _patch_single(monkeypatch, 1)
        assert client.post(f"{PREFIX}/grid", json=sample_config).status_code == 500

    def test_null_surface_pointer_is_500(self, client, sample_config, monkeypatch):
        # status 0 but the C++ side never populated the matrix address -> guarded 500.
        _patch_single(monkeypatch, 0, ptr_value=0)
        resp = client.post(f"{PREFIX}/grid", json=sample_config)
        assert resp.status_code == 500
        assert "address" in resp.json()["detail"].lower()


# ==================================================================
# POST /pricing/chart
# ==================================================================
class TestChart:
    def test_success_returns_png(self, client, sample_config):
        resp = client.post(f"{PREFIX}/chart", json=sample_config)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        # PNG magic number.
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(resp.content) > 1000  # a real rendered figure, not an empty buffer

    def test_requires_auth(self, anon_client, sample_config):
        assert anon_client.post(f"{PREFIX}/chart", json=sample_config).status_code == 401

    def test_invalid_body_is_422(self, client, sample_config):
        sample_config["sigma"] = "bad"
        assert client.post(f"{PREFIX}/chart", json=sample_config).status_code == 422

    def test_license_expired_is_403(self, client, sample_config, monkeypatch):
        _patch_single(monkeypatch, -99)
        assert client.post(f"{PREFIX}/chart", json=sample_config).status_code == 403

    def test_engine_failure_is_500(self, client, sample_config, monkeypatch):
        _patch_single(monkeypatch, 1)
        assert client.post(f"{PREFIX}/chart", json=sample_config).status_code == 500


# ==================================================================
# Cross-cutting: auth is enforced uniformly across JSON-body routes
# ==================================================================
@pytest.mark.parametrize("endpoint", JSON_BODY_ENDPOINTS)
def test_all_json_endpoints_reject_anonymous(anon_client, sample_config, endpoint):
    assert anon_client.post(endpoint, json=sample_config).status_code == 401
