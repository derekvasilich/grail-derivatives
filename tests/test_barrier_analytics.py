"""Tests for the analytic single-barrier pricer (Reiner–Rubinstein) in the validation report.

This is ground-truth used to validate the FDM barrier engine once it lands. Since we can't
lean on the engine yet, the formulas are validated two independent ways:
  1. Limiting behaviour (deep barrier → trivial values; in–out parity).
  2. An independent Brownian-bridge Monte-Carlo continuous-barrier estimate.
"""
import math

import numpy as np
import pytest

from app.api import (
    BarrierType,
    DerivType,
    OptionConfig,
    fdm_price_single,
)
from scripts.generate_validation_report import (
    black_scholes,
    black_scholes_barrier,
    black_scholes_double_barrier,
)

# Base market for the tests.
S, T, SIGMA, R, Q = 100.0, 1.0, 0.3, 0.05, 0.0


def vanilla(k, is_call):
    return black_scholes(S, k, T, SIGMA, R, Q, is_call)["price"]


class TestLimits:
    def test_in_out_parity(self):
        # knock_in + knock_out == vanilla, for every single-barrier flavour.
        for is_call in (True, False):
            for is_down in (True, False):
                k = 100.0
                h = 90.0 if is_down else 110.0
                ki = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call, is_down, is_in=True)
                ko = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call, is_down, is_in=False)
                assert ki + ko == pytest.approx(vanilla(k, is_call), abs=1e-9)

    def test_far_barrier_knockin_is_worthless(self):
        # A down-barrier far below spot is almost never hit → knock-in ≈ 0, knock-out ≈ vanilla.
        k, h = 100.0, 1.0
        ki = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call=True, is_down=True, is_in=True)
        ko = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call=True, is_down=True, is_in=False)
        assert ki == pytest.approx(0.0, abs=1e-3)
        assert ko == pytest.approx(vanilla(k, True), abs=1e-3)

    def test_barrier_at_spot_knocks_in(self):
        # Barrier essentially at spot → knock-in ≈ vanilla, knock-out ≈ 0.
        k, h = 100.0, 99.999
        ki = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call=True, is_down=True, is_in=True)
        ko = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call=True, is_down=True, is_in=False)
        assert ki == pytest.approx(vanilla(k, True), abs=5e-2)
        assert ko == pytest.approx(0.0, abs=5e-2)

    def test_already_breached(self):
        # Spot below a down-barrier: knock-out already dead (0), knock-in already a vanilla.
        ko = black_scholes_barrier(S, 100.0, 110.0, T, SIGMA, R, Q, is_call=True, is_down=True, is_in=False)
        ki = black_scholes_barrier(S, 100.0, 110.0, T, SIGMA, R, Q, is_call=True, is_down=True, is_in=True)
        assert ko == 0.0
        assert ki == pytest.approx(vanilla(100.0, True), abs=1e-12)


def _mc_knockout(k, h, is_call, is_down, n_paths=200_000, n_steps=100, seed=12345):
    """Independent continuous knock-out price via GBM + Brownian-bridge survival correction.

    Between sampled points the probability the path crossed the barrier is known in closed form
    for a Brownian bridge, which makes the continuous-barrier estimate accurate at modest n_steps.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (R - Q - 0.5 * SIGMA * SIGMA) * dt
    vol = SIGMA * math.sqrt(dt)
    logS = np.full(n_paths, math.log(S))
    survive = np.ones(n_paths)
    var = SIGMA * SIGMA * dt
    for _ in range(n_steps):
        prev = logS
        logS = prev + drift + vol * rng.standard_normal(n_paths)
        lp, ln = np.exp(prev), np.exp(logS)
        if is_down:
            crossed_now = (lp <= h) | (ln <= h)
            # P(min of bridge ≤ h | endpoints above h)
            p_cross = np.where((lp > h) & (ln > h),
                               np.exp(-2.0 * np.log(lp / h) * np.log(ln / h) / var), 1.0)
        else:
            crossed_now = (lp >= h) | (ln >= h)
            p_cross = np.where((lp < h) & (ln < h),
                               np.exp(-2.0 * np.log(h / lp) * np.log(h / ln) / var), 1.0)
        p_cross = np.where(crossed_now, 1.0, p_cross)
        survive *= (1.0 - p_cross)
    sT = np.exp(logS)
    payoff = np.maximum(sT - k, 0.0) if is_call else np.maximum(k - sT, 0.0)
    return math.exp(-R * T) * float(np.mean(payoff * survive))


class TestMonteCarloCrossCheck:
    @pytest.mark.parametrize("k,h,is_call,is_down", [
        (100.0, 90.0, True, True),     # down-and-out call (K>H)
        (100.0, 90.0, False, True),    # down-and-out put  (K>H)
        (100.0, 115.0, True, False),   # up-and-out call   (K<H)
        (100.0, 115.0, False, False),  # up-and-out put    (K<H)
        (120.0, 115.0, False, False),  # up-and-out put, deep-ITM (K>H, the tricky branch)
    ])
    def test_knockout_matches_mc(self, k, h, is_call, is_down):
        analytic = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call, is_down, is_in=False)
        mc = _mc_knockout(k, h, is_call, is_down)
        # Independent method → agreement to ~2% (abs floor for MC noise) validates the closed form.
        assert mc == pytest.approx(analytic, rel=0.02, abs=0.1)


def _mc_double_knockout(k, lower, upper, is_call, n_paths=200_000, n_steps=200, seed=777):
    """Independent continuous double-knock-out price via GBM + per-step Brownian-bridge survival."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (R - Q - 0.5 * SIGMA * SIGMA) * dt
    vol = SIGMA * math.sqrt(dt)
    var = SIGMA * SIGMA * dt
    ln_l, ln_u = math.log(lower), math.log(upper)
    logS = np.full(n_paths, math.log(S))
    survive = np.ones(n_paths)
    for _ in range(n_steps):
        prev = logS
        logS = prev + drift + vol * rng.standard_normal(n_paths)
        outside = (prev <= ln_l) | (prev >= ln_u) | (logS <= ln_l) | (logS >= ln_u)
        p_low = np.exp(-2.0 * (prev - ln_l) * (logS - ln_l) / var)   # P(min < lower)
        p_up = np.exp(-2.0 * (ln_u - prev) * (ln_u - logS) / var)    # P(max > upper)
        p_cross = np.where(outside, 1.0, np.clip(p_low + p_up, 0.0, 1.0))
        survive *= (1.0 - p_cross)
    sT = np.exp(logS)
    payoff = np.maximum(sT - k, 0.0) if is_call else np.maximum(k - sT, 0.0)
    return math.exp(-R * T) * float(np.mean(payoff * survive))


class TestDoubleBarrier:
    def test_reduces_to_single_when_one_barrier_is_far(self):
        # Upper barrier far away → double knock-out ≈ single down-and-out (validated R–R).
        k, lower = 100.0, 85.0
        dko = black_scholes_double_barrier(S, k, lower, 1.0e6, T, SIGMA, R, Q, is_call=True, is_in=False)
        ddo = black_scholes_barrier(S, k, lower, T, SIGMA, R, Q, is_call=True, is_down=True, is_in=False)
        assert dko == pytest.approx(ddo, abs=1e-3)
        # Lower barrier near zero → double knock-out ≈ single up-and-out.
        k, upper = 100.0, 115.0
        dko2 = black_scholes_double_barrier(S, k, 1.0e-6, upper, T, SIGMA, R, Q, is_call=True, is_in=False)
        duo = black_scholes_barrier(S, k, upper, T, SIGMA, R, Q, is_call=True, is_down=False, is_in=False)
        assert dko2 == pytest.approx(duo, abs=1e-3)

    def test_in_out_parity(self):
        for is_call in (True, False):
            ko = black_scholes_double_barrier(S, 100.0, 85.0, 120.0, T, SIGMA, R, Q, is_call, is_in=False)
            ki = black_scholes_double_barrier(S, 100.0, 85.0, 120.0, T, SIGMA, R, Q, is_call, is_in=True)
            assert ko + ki == pytest.approx(vanilla(100.0, is_call), abs=1e-9)

    def test_wide_corridor_approaches_vanilla(self):
        ko = black_scholes_double_barrier(S, 100.0, 1e-4, 1e6, T, SIGMA, R, Q, is_call=True, is_in=False)
        assert ko == pytest.approx(vanilla(100.0, True), abs=1e-2)

    def test_already_breached(self):
        assert black_scholes_double_barrier(S, 100.0, 105.0, 120.0, T, SIGMA, R, Q, True, is_in=False) == 0.0

    @pytest.mark.parametrize("k,lower,upper,is_call", [
        (100.0, 85.0, 120.0, True),    # double knock-out call, corridor around spot
        (100.0, 85.0, 120.0, False),   # double knock-out put
    ])
    def test_double_knockout_matches_mc(self, k, lower, upper, is_call):
        analytic = black_scholes_double_barrier(S, k, lower, upper, T, SIGMA, R, Q, is_call, is_in=False)
        mc = _mc_double_knockout(k, lower, upper, is_call)
        assert mc == pytest.approx(analytic, rel=0.02, abs=0.1)


# ============================================================================
# FDM engine tests. These exercise the real finite-difference barrier solver
# (x-space, continuous monitoring with the barrier pinned on a grid node) and
# compare against the Reiner-Rubinstein / Kunitomo-Ikeda closed forms above.
# The closed forms are continuous-barrier, so the FDM error is pure
# discretisation and shrinks with the grid; the tolerances reflect a fine grid.
# ============================================================================

# A fine, fast grid: dx ~ h / max(S, K) ~ 0.0025 in log-price, 600 time steps.
_FDM_H = 0.25
_FDM_TN = 600
_FDM_TOL = 0.05  # a few cents; observed worst case ~0.015


def _fdm_single_barrier(k, h_level, is_call, is_down, is_in):
    """Price one single-barrier flavour through the FDM engine."""
    if is_call:
        deriv = DerivType.BarrierInCall if is_in else DerivType.BarrierOutCall
    else:
        deriv = DerivType.BarrierInPut if is_in else DerivType.BarrierOutPut
    barrier = BarrierType.DownAndOut if is_down else BarrierType.UpAndOut
    cfg = OptionConfig(
        time=T, h=_FDM_H, r=R, sigma=SIGMA, s=S, k=k, q=Q,
        b_low=(h_level if is_down else 0.0),
        b_up=(0.0 if is_down else h_level),
        deriv=deriv, barrier=barrier, Tn=_FDM_TN,
    )
    status, greeks, _ = fdm_price_single(cfg)
    assert status == 0
    return greeks.price


def _fdm_double_barrier(k, lower, upper, is_call, is_in):
    """Price one double-barrier flavour through the FDM engine."""
    if is_call:
        deriv = DerivType.DblBarrierInCall if is_in else DerivType.DblBarrierOutCall
    else:
        deriv = DerivType.DblBarrierInPut if is_in else DerivType.DblBarrierOutPut
    cfg = OptionConfig(
        time=T, h=_FDM_H, r=R, sigma=SIGMA, s=S, k=k, q=Q,
        b_low=lower, b_up=upper, deriv=deriv, Tn=_FDM_TN,
    )
    status, greeks, _ = fdm_price_single(cfg)
    assert status == 0
    return greeks.price


class TestFDMSingleBarrier:
    @pytest.mark.parametrize("k,h,is_call,is_down", [
        (100.0, 90.0, True, True),     # down-and-out call
        (100.0, 90.0, False, True),    # down-and-out put
        (100.0, 115.0, True, False),   # up-and-out call
        (100.0, 115.0, False, False),  # up-and-out put
        (120.0, 115.0, False, False),  # up-and-out put, deep-ITM (K > H)
    ])
    def test_knockout_matches_analytic(self, k, h, is_call, is_down):
        fdm = _fdm_single_barrier(k, h, is_call, is_down, is_in=False)
        analytic = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call, is_down, is_in=False)
        assert fdm == pytest.approx(analytic, abs=_FDM_TOL)

    @pytest.mark.parametrize("k,h,is_call,is_down", [
        (100.0, 90.0, True, True),     # down-and-in call (priced via parity)
        (100.0, 115.0, False, False),  # up-and-in put
    ])
    def test_knockin_matches_analytic(self, k, h, is_call, is_down):
        fdm = _fdm_single_barrier(k, h, is_call, is_down, is_in=True)
        analytic = black_scholes_barrier(S, k, h, T, SIGMA, R, Q, is_call, is_down, is_in=True)
        assert fdm == pytest.approx(analytic, abs=_FDM_TOL)

    @pytest.mark.parametrize("is_call,is_down", [
        (True, True), (False, True), (True, False), (False, False),
    ])
    def test_in_out_parity_holds_in_engine(self, is_call, is_down):
        # The engine prices knock-ins by parity, so this is exact up to FP noise.
        k = 100.0
        h = 90.0 if is_down else 110.0
        ko = _fdm_single_barrier(k, h, is_call, is_down, is_in=False)
        ki = _fdm_single_barrier(k, h, is_call, is_down, is_in=True)
        vanilla_price = black_scholes(S, k, T, SIGMA, R, Q, is_call)["price"]
        assert ko + ki == pytest.approx(vanilla_price, abs=_FDM_TOL)


class TestFDMDoubleBarrier:
    @pytest.mark.parametrize("k,lower,upper,is_call", [
        (100.0, 85.0, 120.0, True),    # double knock-out call
        (100.0, 85.0, 120.0, False),   # double knock-out put
    ])
    def test_knockout_matches_analytic(self, k, lower, upper, is_call):
        fdm = _fdm_double_barrier(k, lower, upper, is_call, is_in=False)
        analytic = black_scholes_double_barrier(S, k, lower, upper, T, SIGMA, R, Q, is_call, is_in=False)
        assert fdm == pytest.approx(analytic, abs=_FDM_TOL)

    @pytest.mark.parametrize("is_call", [True, False])
    def test_in_out_parity_holds_in_engine(self, is_call):
        ko = _fdm_double_barrier(100.0, 85.0, 120.0, is_call, is_in=False)
        ki = _fdm_double_barrier(100.0, 85.0, 120.0, is_call, is_in=True)
        vanilla_price = black_scholes(S, 100.0, T, SIGMA, R, Q, is_call)["price"]
        assert ko + ki == pytest.approx(vanilla_price, abs=_FDM_TOL)
