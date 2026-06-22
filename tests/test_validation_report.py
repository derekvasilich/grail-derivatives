"""Tests for the engine validation-report generator (scripts/generate_validation_report.py)."""
import math

import pytest

from scripts.generate_validation_report import black_scholes, render_markdown, run_validation


class TestAnalyticBSM:
    def test_matches_known_reference(self):
        # Same case as the convergence test's analytic anchor.
        result = black_scholes(100.0, 100.0, 1.0, 0.2, 0.1, 0.0, is_call=True)
        assert result["price"] == pytest.approx(13.26967658, abs=1e-6)

    def test_put_call_parity(self):
        s, k, t, sigma, r, q = 100.0, 110.0, 1.0, 0.5, 0.1, 0.0
        call = black_scholes(s, k, t, sigma, r, q, is_call=True)["price"]
        put = black_scholes(s, k, t, sigma, r, q, is_call=False)["price"]
        # C - P = S e^{-qT} - K e^{-rT}
        assert call - put == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-9)


class TestReportGeneration:
    @pytest.fixture
    def report(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VALIDATION_OUT", str(tmp_path))
        return run_validation(quick=True)

    def test_rigorous_gates_pass(self, report):
        # The validated comparisons (analytic + American + convergence + determinism) must hold.
        for gate in ("euro_price", "euro_delta", "euro_gamma", "euro_vega",
                     "american_price", "convergence", "determinism"):
            assert report["gates"][gate] is True, f"gate {gate} regressed"
        assert report["passed"] is True

    def test_vega_validated_against_analytic(self, report):
        # Vega is now computed by the engine and checked against closed-form BSM vega.
        assert report["sections"]["accuracy"]["european_vs_analytic"]["max_abs_err"]["vega"] <= 2e-1

    def test_convergence_is_second_order(self, report):
        assert 1.90 <= report["sections"]["convergence"]["slope"] <= 2.10

    def test_determinism_holds(self, report):
        assert report["sections"]["determinism"]["identical"] is True

    def test_chart_artifact_written(self, report, tmp_path):
        assert (tmp_path / "convergence.png").exists()

    def test_markdown_renders(self, report):
        md = render_markdown(report)
        assert "# Grail Derivatives — Engine Validation Report" in md
        assert "European accuracy vs. closed-form Black–Scholes" in md
        # Per-scenario appendices (European + American + Bermudan) present for thoroughness.
        assert "Appendix A — European accuracy, per scenario" in md
        assert "vega (fdm / analytic)" in md
        assert "Appendix B — American accuracy, per scenario" in md
        assert "Appendix C — Bermudan accuracy, per scenario" in md

    def test_samples_carry_vega(self, report):
        samples = report["sections"]["accuracy"]["european_vs_analytic"]["samples"]
        assert samples, "expected per-point European samples"
        for row in samples:
            assert {"fdm_vega", "analytic_vega", "vega_abs_err"} <= row.keys()
