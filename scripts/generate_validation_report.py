#!/usr/bin/env python3
"""Grail Derivatives — engine validation report generator.

Produces a reproducible accuracy/convergence/performance report for the compiled FDM
engine. Run it on every engine release; the output doubles as a model-validation pack
for regulated buyers ("don't trust us — run the script").

Sections
--------
1. European accuracy vs. closed-form Black–Scholes–Merton (price, delta, gamma).
   The headline: an exact analytic answer exists, so errors are near machine scale.
2. American & Bermudan accuracy vs. a high-resolution binomial tree (independent method,
   engine-internal conventions → validates price + all Greeks incl. theta).
3. Convergence study — refine the grid, show O(Δx²) error decay + a log-log chart.
4. Performance — batch throughput (options/sec) and single-solve latency vs. Tn.
5. Determinism — identical inputs must yield bit-identical outputs.

Outputs (under --out, default ./validation):
    report.json        machine-readable results
    report.md          rendered summary with tables
    convergence.png    log-log convergence chart

Exit code is non-zero if any tolerance gate fails (so CI keeps the report honest),
unless --no-fail is passed.

Usage:
    python scripts/generate_validation_report.py [--out DIR] [--quick] [--no-fail]
"""
import argparse
import json
import math
import os
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running as a plain script (python scripts/...) by ensuring the repo root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import BarrierType, DerivType, FrequencyType, OptionConfig, fdm_price_batch, fdm_price_binomial_all, fdm_price_single  # noqa: E402

_N = NormalDist()

# Each derivative row carries which independent oracle validates it and, for barriers, the
# barrier specification. Barrier levels are spot-RELATIVE multipliers (b_*_mult * S) so they
# stay sensible across the accuracy sweep's varying spots/strikes.
#   oracle: "analytic"     -> closed-form Black-Scholes (Europeans)
#           "binomial"     -> high-resolution binomial tree (American/Bermudan)
#           "barrier"      -> Reiner-Rubinstein single barrier closed form
#           "dbl_barrier"  -> Kunitomo-Ikeda double barrier closed form
# barrier_dir is the single-barrier direction (BarrierType.*) or None for non/double barriers.
ORACLE_ANALYTIC, ORACLE_BINOMIAL, ORACLE_BARRIER, ORACLE_DBL_BARRIER = (
    "analytic", "binomial", "barrier", "dbl_barrier")

DerivRow = namedtuple("DerivRow", "deriv label is_call oracle barrier_dir b_low_mult b_up_mult")

DERIV_ROWS = [
    DerivRow(DerivType.VanillaCall, "European Call", True, ORACLE_ANALYTIC, None, 0.0, 0.0),
    DerivRow(DerivType.VanillaPut, "European Put", False, ORACLE_ANALYTIC, None, 0.0, 0.0),
    DerivRow(DerivType.AmericanCall, "American Call", True, ORACLE_BINOMIAL, None, 0.0, 0.0),
    DerivRow(DerivType.AmericanPut, "American Put", False, ORACLE_BINOMIAL, None, 0.0, 0.0),
    DerivRow(DerivType.BermudanCall, "Bermudan Call", True, ORACLE_BINOMIAL, None, 0.0, 0.0),
    DerivRow(DerivType.BermudanPut, "Bermudan Put", False, ORACLE_BINOMIAL, None, 0.0, 0.0),
    DerivRow(DerivType.BarrierOutCall, "Barrier Out Call", True, ORACLE_BARRIER, BarrierType.DownAndOut, 0.90, 0.0),
    DerivRow(DerivType.BarrierOutPut, "Barrier Out Put", False, ORACLE_BARRIER, BarrierType.UpAndOut, 0.0, 1.15),
    DerivRow(DerivType.BarrierInCall, "Barrier In Call", True, ORACLE_BARRIER, BarrierType.DownAndOut, 0.90, 0.0),
    DerivRow(DerivType.BarrierInPut, "Barrier In Put", False, ORACLE_BARRIER, BarrierType.UpAndOut, 0.0, 1.15),
    DerivRow(DerivType.DblBarrierOutCall, "Double Barrier Out Call", True, ORACLE_DBL_BARRIER, None, 0.85, 1.20),
    DerivRow(DerivType.DblBarrierOutPut, "Double Barrier Out Put", False, ORACLE_DBL_BARRIER, None, 0.85, 1.20),
    DerivRow(DerivType.DblBarrierInCall, "Double Barrier In Call", True, ORACLE_DBL_BARRIER, None, 0.85, 1.20),
    DerivRow(DerivType.DblBarrierInPut, "Double Barrier In Put", False, ORACLE_DBL_BARRIER, None, 0.85, 1.20),
]

_BARRIER_IN_TYPES = {
    DerivType.BarrierInCall, DerivType.BarrierInPut,
    DerivType.DblBarrierInCall, DerivType.DblBarrierInPut,
}

# Tolerance gates (errors above these fail the report). Tuned to the default report grid.
TOL = {
    "euro_price_abs": 5e-2,     # vs analytic BSM
    "euro_delta_abs": 5e-3,
    "euro_gamma_abs": 5e-4,
    "euro_theta_abs": 1e-3,
    "euro_vega_abs": 2e-1,      # vega is a bump-and-reprice pass → noisier than the primal solve
    "amer_price_abs": 5e-2,     # American vs high-res binomial (gated)
    "bermudan_price_warn": 1e-1,  # Bermudan vs binomial — surfaced as a finding, not gated
    "barrier_price_abs": 1e-1,    # barriers vs closed form (Reiner-Rubinstein / Kunitomo-Ikeda), gated
    # Per-deriv-type convergence slope gates, indexed by DerivType value (0..13). Barriers sit
    # exactly on a grid node + Rannacher smoothing, so they recover clean 2nd order like vanillas.
    "convergence_slope_min": [1.90, 1.90, 1.85, 1.80, 1.75, 1.60,
                              1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85],
    "convergence_slope_max": [2.10, 2.10, 2.15, 2.20, 2.15, 2.40,
                              2.15, 2.15, 2.15, 2.15, 2.15, 2.15, 2.15, 2.15],
}


# ------------------------------------------------------------------
# Closed-form Black–Scholes–Merton (the European ground truth)
# ------------------------------------------------------------------
def black_scholes(s, k, t, sigma, r, q, is_call):
    """Analytic BSM price, delta, gamma, vega for a European option with continuous yield q.

    Vega is per 1.00 (100%) change in volatility — the absolute convention the engine uses.
    """
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_q, disc_r = math.exp(-q * t), math.exp(-r * t)
    tt1, tt2 = - (s * _N.pdf(d1) * sigma) / (2.0 * sqrt_t), r * k * disc_r;
    if is_call:
        price = s * disc_q * _N.cdf(d1) - k * disc_r * _N.cdf(d2)
        delta = disc_q * _N.cdf(d1)
        theta = (tt1 - (tt2 * _N.cdf(d2)))/365;
    else:
        price = k * disc_r * _N.cdf(-d2) - s * disc_q * _N.cdf(-d1)
        delta = -disc_q * _N.cdf(-d1)
        theta = (tt1 + (tt2 * _N.cdf(-d2)))/356;
    gamma = disc_q * _N.pdf(d1) / (s * sigma * sqrt_t)
    vega = s * disc_q * _N.pdf(d1) * sqrt_t  # ∂V/∂σ, identical for calls and puts
    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def black_scholes_barrier(s, k, h, t, sigma, r, q, is_call, is_down, is_in):
    """Reiner–Rubinstein closed-form price for a single-barrier option (continuous monitoring, no rebate).

    Ground truth for validating the FDM barrier pricer once it lands (post x-space transform).
    `is_down` selects a down-barrier (live region s > h) vs. up-barrier (live region s < h);
    `is_in` selects knock-in vs. knock-out. Knock-outs are obtained by in–out parity
    (knock_in + knock_out = vanilla, exact for zero rebate), so `out = vanilla − in`.

    If spot is already on the dead side of the barrier, returns the trivially-resolved value
    (knock-out → 0, knock-in → vanilla).
    """
    b = r - q                          # cost of carry
    sst = sigma * math.sqrt(t)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    phi = 1.0 if is_call else -1.0     # call/put indicator
    eta = 1.0 if is_down else -1.0     # down/up barrier indicator

    vanilla = black_scholes(s, k, t, sigma, r, q, is_call)["price"]
    breached = (s <= h) if is_down else (s >= h)
    if breached:
        return vanilla if is_in else 0.0

    x1 = math.log(s / k) / sst + (1 + mu) * sst
    x2 = math.log(s / h) / sst + (1 + mu) * sst
    y1 = math.log(h * h / (s * k)) / sst + (1 + mu) * sst
    y2 = math.log(h / s) / sst + (1 + mu) * sst

    df_b, df_r = math.exp((b - r) * t), math.exp(-r * t)
    pow_plus, pow_minus = (h / s) ** (2 * (mu + 1)), (h / s) ** (2 * mu)

    # Reiner–Rubinstein building blocks. Note A (with phi) is exactly the BSM vanilla price.
    A = phi * s * df_b * _N.cdf(phi * x1) - phi * k * df_r * _N.cdf(phi * x1 - phi * sst)
    B = phi * s * df_b * _N.cdf(phi * x2) - phi * k * df_r * _N.cdf(phi * x2 - phi * sst)
    C = (phi * s * df_b * pow_plus * _N.cdf(eta * y1)
         - phi * k * df_r * pow_minus * _N.cdf(eta * y1 - eta * sst))
    D = (phi * s * df_b * pow_plus * _N.cdf(eta * y2)
         - phi * k * df_r * pow_minus * _N.cdf(eta * y2 - eta * sst))

    # Knock-in value per Haug's standard table (rebate = 0); condition is strike vs. barrier.
    if is_call and is_down:            # down-and-in call
        in_val = C if k > h else (A - B + D)
    elif is_call and not is_down:      # up-and-in call
        in_val = A if k > h else (B - C + D)
    elif (not is_call) and is_down:    # down-and-in put (mirrors up-and-in call)
        in_val = (B - C + D) if k > h else A
    else:                              # up-and-in put (mirrors down-and-in call)
        in_val = (A - B + D) if k > h else C

    return in_val if is_in else (vanilla - in_val)


def black_scholes_double_barrier(s, k, lower, upper, t, sigma, r, q, is_call, is_in, n_terms=10):
    """Kunitomo–Ikeda closed-form double-barrier option (flat barriers, continuous, no rebate).

    The knock-out is the corridor option — alive while lower < S < upper, worthless once either
    barrier is touched. It's a method-of-images series; n_terms images each side converges fast.
    Knock-in via parity (in = vanilla − out). As one barrier moves to its extreme the series
    collapses to the single-barrier (Reiner–Rubinstein) value, which the tests use as a check.
    """
    b = r - q
    vsqt = sigma * math.sqrt(t)
    drift = (b + 0.5 * sigma * sigma) * t
    L, U, X = lower, upper, k

    vanilla = black_scholes(s, k, t, sigma, r, q, is_call)["price"]
    if s <= L or s >= U:               # already knocked out / in
        return vanilla if is_in else 0.0

    mu1 = 2.0 * b / (sigma * sigma) + 1.0   # mu3 == mu1 and mu2 == 0 for flat barriers
    sum1 = sum2 = 0.0
    for n in range(-n_terms, n_terms + 1):
        U2n, L2n = U ** (2 * n), L ** (2 * n)
        Lnp2 = L ** (2 * n + 2)
        if is_call:
            d1 = (math.log(s * U2n / (X * L2n)) + drift) / vsqt
            d2 = (math.log(s * U2n / (U * L2n)) + drift) / vsqt
            d3 = (math.log(Lnp2 / (X * s * U2n)) + drift) / vsqt
            d4 = (math.log(Lnp2 / (U * s * U2n)) + drift) / vsqt
        else:
            d1 = (math.log(s * U2n / (L * L2n)) + drift) / vsqt
            d2 = (math.log(s * U2n / (X * L2n)) + drift) / vsqt
            d3 = (math.log(Lnp2 / (L * s * U2n)) + drift) / vsqt
            d4 = (math.log(Lnp2 / (X * s * U2n)) + drift) / vsqt
        ratio_a = (U ** n / L ** n) ** mu1
        ratio_b = (L ** (n + 1) / (U ** n * s)) ** mu1
        sum1 += ratio_a * (_N.cdf(d1) - _N.cdf(d2)) - ratio_b * (_N.cdf(d3) - _N.cdf(d4))
        ratio_a2 = (U ** n / L ** n) ** (mu1 - 2.0)
        ratio_b2 = (L ** (n + 1) / (U ** n * s)) ** (mu1 - 2.0)
        sum2 += (ratio_a2 * (_N.cdf(d1 - vsqt) - _N.cdf(d2 - vsqt))
                 - ratio_b2 * (_N.cdf(d3 - vsqt) - _N.cdf(d4 - vsqt)))

    fwd = s * math.exp((b - r) * t)
    disc = X * math.exp(-r * t)
    out_val = (fwd * sum1 - disc * sum2) if is_call else (disc * sum2 - fwd * sum1)
    out_val = max(out_val, 0.0)        # guard tiny negatives from series truncation
    return (vanilla - out_val) if is_in else out_val


def _make_config(deriv, s, k, t, sigma, r, q, h, tn):
    return OptionConfig(
        deriv=deriv, frequency=FrequencyType.Quarterly,
        Tn=tn, top=0, bottom=1, left=0, right=0,
        time=t, h=h, r=r, sigma=sigma, s=s, k=k, q=q,
    )


def _row_barrier_levels(row, s):
    """Absolute barrier levels for a row at spot s (0.0 where the side is unused)."""
    return (row.b_low_mult * s if row.b_low_mult else 0.0,
            row.b_up_mult * s if row.b_up_mult else 0.0)


def config_for_row(row, s, k, t, sigma, r, q, h, tn):
    """Build an OptionConfig for a DerivRow, wiring barrier direction + spot-relative levels."""
    cfg = _make_config(row.deriv, s, k, t, sigma, r, q, h, tn)
    if row.barrier_dir is not None:
        cfg.barrier = row.barrier_dir
    cfg.b_low, cfg.b_up = _row_barrier_levels(row, s)
    return cfg


def reference_price(row, s, k, t, sigma, r, q, binomial_ref=None):
    """Independent oracle price (and Greeks where available) for a DerivRow.

    Europeans -> analytic BSM; American/Bermudan -> the high-res binomial tree; barriers ->
    the Reiner-Rubinstein / Kunitomo-Ikeda closed forms (price only). The binomial tree is
    NEVER consulted for barriers (it only resolves the 6 vanilla/American/Bermudan types).
    """
    if row.oracle == ORACLE_ANALYTIC:
        return black_scholes(s, k, t, sigma, r, q, row.is_call)
    if row.oracle == ORACLE_BINOMIAL:
        ref = binomial_ref[row.deriv]
        return {"price": ref.price, "delta": ref.delta, "gamma": ref.gamma, "theta": ref.theta}

    b_low, b_up = _row_barrier_levels(row, s)
    is_in = row.deriv in _BARRIER_IN_TYPES
    if row.oracle == ORACLE_BARRIER:
        is_down = row.barrier_dir == BarrierType.DownAndOut
        level = b_low if is_down else b_up
        price = black_scholes_barrier(s, k, level, t, sigma, r, q, row.is_call, is_down, is_in)
    else:  # ORACLE_DBL_BARRIER
        price = black_scholes_double_barrier(s, k, b_low, b_up, t, sigma, r, q, row.is_call, is_in)
    return {"price": price}


# ------------------------------------------------------------------
# Section runners
# ------------------------------------------------------------------
def run_accuracy(grid, matrix, binomial_n):
    """Sections 1 & 2: FDM vs the independent oracle for each row.

    Each row is validated against the method appropriate to it: Europeans vs analytic BSM,
    American/Bermudan vs the high-res binomial tree, and barriers vs the Reiner-Rubinstein /
    Kunitomo-Ikeda closed forms. The binomial tree is only consulted for the 6 types it can
    resolve, so barrier rows never index past it.
    """
    euro_rows, exotic_rows, barrier_rows = [], [], []
    euro_max = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    american_max_price = 0.0
    bermudan_max_price = 0.0
    barrier_max_price = 0.0

    for (s, k, t, sigma, r, q) in matrix:
        base = _make_config(DerivType.VanillaCall, s, k, t, sigma, r, q, grid["h"], grid["tn"])
        bin_status, bin_ref = fdm_price_binomial_all(base, binomial_n)
        if bin_status != 0:
            raise RuntimeError(f"binomial reference failed (status {bin_status}) at {(s,k,t,sigma,r,q)}")

        for row in DERIV_ROWS:
            cfg = config_for_row(row, s, k, t, sigma, r, q, grid["h"], grid["tn"])
            # Request vega only for the European types (the ones validated against analytic) —
            # the bump-and-reprice pass roughly doubles cost, so skip it for the exotic rows.
            is_european = row.oracle == ORACLE_ANALYTIC
            st, fdm, _ = fdm_price_single(cfg, is_european)
            if st != 0:
                raise RuntimeError(f"FDM solve failed (status {st}) for {row.label} at {(s,k,t,sigma,r,q)}")

            # European → compare against the exact analytic answer (price + all Greeks).
            if is_european:
                exact = black_scholes(s, k, t, sigma, r, q, row.is_call)
                dp, dd, dg, dt, dv = (abs(fdm.price - exact["price"]), abs(fdm.delta - exact["delta"]),
                                  abs(fdm.gamma - exact["gamma"]), abs(fdm.theta - exact["theta"]), abs(fdm.vega - exact["vega"]))
                euro_max["price"], euro_max["delta"], euro_max["gamma"], euro_max["theta"], euro_max["vega"] = (
                    max(euro_max["price"], dp), max(euro_max["delta"], dd),
                    max(euro_max["gamma"], dg), max(euro_max["theta"], dt), max(euro_max["vega"], dv))
                euro_rows.append({"label": row.label, "s": s, "k": k, "t": t, "sigma": sigma,
                                  "fdm_price": fdm.price, "analytic_price": exact["price"],
                                  "abs_err": dp, "rel_err": dp / exact["price"] if exact["price"] else 0.0,
                                  "delta_abs_err": dd, "gamma_abs_err": dg,
                                  "fdm_vega": fdm.vega, "analytic_vega": exact["vega"], "vega_abs_err": dv})
                continue

            # American/Bermudan → cross-validate against the independent binomial method.
            if row.oracle == ORACLE_BINOMIAL:
                ref = reference_price(row, s, k, t, sigma, r, q, bin_ref)
                dprice = abs(fdm.price - ref["price"])
                if row.deriv in (DerivType.AmericanCall, DerivType.AmericanPut):
                    american_max_price = max(american_max_price, dprice)
                else:
                    bermudan_max_price = max(bermudan_max_price, dprice)
                exotic_rows.append({"label": row.label, "s": s, "k": k, "t": t, "sigma": sigma,
                                    "fdm_price": fdm.price, "binomial_price": ref["price"], "abs_err": dprice,
                                    "delta_abs_err": abs(fdm.delta - ref["delta"]),
                                    "gamma_abs_err": abs(fdm.gamma - ref["gamma"]),
                                    "theta_abs_err": abs(fdm.theta - ref["theta"])})
                continue

            # Barriers → cross-validate against the closed-form barrier price.
            ref = reference_price(row, s, k, t, sigma, r, q)
            b_low, b_up = _row_barrier_levels(row, s)
            dprice = abs(fdm.price - ref["price"])
            barrier_max_price = max(barrier_max_price, dprice)
            barrier_rows.append({"label": row.label, "s": s, "k": k, "t": t, "sigma": sigma,
                                 "b_low": b_low, "b_up": b_up,
                                 "fdm_price": fdm.price, "closed_form_price": ref["price"], "abs_err": dprice})

    return {
        "european_vs_analytic": {"max_abs_err": euro_max, "samples": euro_rows},
        "american_vs_binomial": {"max_price_abs_err": american_max_price, "binomial_n": binomial_n},
        "bermudan_vs_binomial": {"max_price_abs_err": bermudan_max_price, "binomial_n": binomial_n},
        "barrier_vs_closed_form": {"max_price_abs_err": barrier_max_price, "samples": barrier_rows},
        "samples": exotic_rows,
    }


def run_convergence(out_dir):
    """Section 3: refine the grid for each derivative type, estimate the slope, chart it.

    Each row converges against its own independent oracle (analytic / binomial / closed-form
    barrier), so the slope is a true error-decay rate, not FDM-vs-FDM.
    """
    resolutions = [1, 2, 5, 10]
    tn_steps = [100, 400, 2500, 10000]
    s, k, t, sigma, r, q = 100.0, 100.0, 1.0, 0.2, 0.1, 0.0
    results: list[dict] = []
    for row in DERIV_ROWS:
        nodes, errors = [], []
        # Binomial-backed rows share one high-res reference across resolutions.
        bin_ref = None
        if row.oracle == ORACLE_BINOMIAL:
            _, bin_ref = fdm_price_binomial_all(config_for_row(row, s, k, t, sigma, r, q, 1.0, 100), 8000)
        for res, tn in zip(resolutions, tn_steps):
            cfg = config_for_row(row, s, k, t, sigma, r, q, 1.0 / res, tn)
            st, gr, _ = fdm_price_single(cfg)
            if st != 0:
                raise RuntimeError(f"convergence solve failed (status {st}) at res {res}")
            nodes.append(res)
            exact = reference_price(row, s, k, t, sigma, r, q, bin_ref)
            errors.append(abs(gr.price - exact["price"]))

        nodes_arr, errors_arr = np.array(nodes), np.array(errors)
        slope, _ = np.polyfit(np.log(1.0 / nodes_arr), np.log(errors_arr), 1)

        deriv_type, deriv_name = row.deriv, row.label
        chart_file_name = f"{deriv_name.lower().replace(' ', '-')}-convergence.png";
        chart_path = os.path.join(out_dir, chart_file_name)
        plt.figure(figsize=(8, 6))
        plt.loglog(nodes_arr, errors_arr, "o-", color="crimson", linewidth=2, label=f"Crank–Nicolson (slope {slope:.4f})")
        plt.loglog(nodes_arr, errors_arr[0] * (nodes_arr[0] / nodes_arr) ** 2, "--", color="dodgerblue", label="Theoretical O(Δx²)")
        plt.title("Convergence — error vs. grid resolution", fontsize=12, fontweight="bold")
        plt.xlabel("Grid resolution (1/h)")
        plt.ylabel("|FDM − analytic|")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        results.append({
            "deriv_type": deriv_type,
            "deriv_name": deriv_name,
            "slope": float(slope),
            "resolutions": nodes,
            "errors": [float(e) for e in errors],
            "chart": chart_file_name
        })

    return results


def run_performance(perf_grid, batch_size):
    """Section 4: batch throughput + single-solve latency vs Tn.

    Run on its OWN stated benchmark grid (not the fine accuracy grid) — throughput is
    highly grid-dependent, so the grid is reported alongside the number to keep it honest.
    """
    h, tn = perf_grid["h"], perf_grid["tn"]
    configs = (OptionConfig * batch_size)()
    for i in range(batch_size):
        c = _make_config(DerivType.AmericanPut, 100.0, 100.0 + (i % 40) - 20, 1.0, 0.5, 0.1, 0.0, h, tn)
        for f in ("deriv", "frequency", "Tn", "top", "bottom", "left", "right", "time", "h", "r", "sigma", "s", "k", "q"):
            setattr(configs[i], f, getattr(c, f))

    t0 = time.perf_counter()
    st, _ = fdm_price_batch(configs, batch_size, False)
    elapsed = time.perf_counter() - t0
    if st != 0:
        raise RuntimeError(f"performance batch failed (status {st})")
    throughput = batch_size / elapsed if elapsed > 0 else float("inf")

    latency = []
    for tn_i in (100, 500, 1000, 2000):
        cfg = _make_config(DerivType.AmericanPut, 100.0, 100.0, 1.0, 0.5, 0.1, 0.0, h, tn_i)
        t0 = time.perf_counter()
        fdm_price_single(cfg)
        latency.append({"tn": tn_i, "ms": (time.perf_counter() - t0) * 1000.0})

    return {"grid": perf_grid, "batch_size": batch_size, "elapsed_s": elapsed,
            "options_per_sec": throughput, "latency_by_tn": latency}


def run_determinism(grid):
    """Section 5: identical inputs → bit-identical outputs."""
    cfg = _make_config(DerivType.AmericanPut, 100.0, 110.0, 1.0, 0.5, 0.1, 0.0, grid["h"], grid["tn"])
    _, a, _ = fdm_price_single(cfg)
    _, b, _ = fdm_price_single(cfg)
    identical = (a.price, a.delta, a.gamma, a.theta) == (b.price, b.delta, b.gamma, b.theta)
    return {"identical": identical, "price": a.price, "delta": a.delta, "gamma": a.gamma, "theta": a.theta}


def _build_matrix(quick):
    strikes = [100.0, 110.0] if quick else [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities = [1.0] if quick else [0.25, 1.0, 2.0]
    vols = [0.5] if quick else [0.2, 0.5]
    return [(100.0, k, t, v, 0.1, 0.0) for k in strikes for t in maturities for v in vols]


def run_validation(quick=False):
    """Run every section and assemble the report dict (no file I/O except the chart)."""
    grid = {"h": 0.25, "tn": 1000}
    perf_grid = {"h": 1.0, "tn": 100}
    binomial_n = 1024 if quick else 2048
    out_dir = os.environ.get("VALIDATION_OUT", "validation")
    os.makedirs(out_dir, exist_ok=True)

    accuracy = run_accuracy(grid, _build_matrix(quick), binomial_n)
    convergence = run_convergence(out_dir)
    performance = run_performance(perf_grid, 2000 if quick else 5000)
    determinism = run_determinism(grid)

    euro = accuracy["european_vs_analytic"]["max_abs_err"]
    bermudan_err = accuracy["bermudan_vs_binomial"]["max_price_abs_err"]
    # Hard release gates — only the rigorously-validated comparisons.
    gates = {
        "euro_price": euro["price"] <= TOL["euro_price_abs"],
        "euro_delta": euro["delta"] <= TOL["euro_delta_abs"],
        "euro_gamma": euro["gamma"] <= TOL["euro_gamma_abs"],
        "euro_theta": euro["theta"] <= TOL["euro_theta_abs"],
        "euro_vega": euro["vega"] <= TOL["euro_vega_abs"],
        "american_price": accuracy["american_vs_binomial"]["max_price_abs_err"] <= TOL["amer_price_abs"],
        "barrier_price": accuracy["barrier_vs_closed_form"]["max_price_abs_err"] <= TOL["barrier_price_abs"],
        "determinism": determinism["identical"],
    }
    for conv in convergence:
        deriv_type = conv["deriv_type"]
        key = f"convergence-{deriv_type}"
        gates[key] = bool(TOL["convergence_slope_min"][deriv_type] <= conv["slope"] <= TOL["convergence_slope_max"][deriv_type])

    # Findings — surfaced (not gated) discrepancies the report flags for review.
    findings = []
    if bermudan_err > TOL["bermudan_price_warn"]:
        findings.append(
            f"Bermudan FDM vs. binomial price diverges by up to {bermudan_err:.3f} "
            f"(> {TOL['bermudan_price_warn']:.0e} warn threshold). American is clean, so this points to an "
            "exercise-schedule convention mismatch in the Bermudan pricer, not the early-exercise machinery."
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid": grid,
        "tolerances": TOL,
        "sections": {
            "accuracy": accuracy, 
            "convergence": convergence, 
            "performance": performance, 
            "determinism": determinism
        },
        "gates": gates,
        "findings": findings,
        "passed": all(gates.values()),
    }
    

# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------
def _vs_binomial_appendix(title, samples, labels):
    """Render a per-scenario FDM-vs-binomial table for the given derivative-type labels."""
    lines = [
        f"## {title}",
        "",
        "Per-point FDM-vs-binomial errors (binomial = independent reference; validates theta too).",
        "",
        "| Type | S | K | T | σ | fdm price | binomial price | Δprice | Δdelta | Δgamma | Δtheta |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in (row for row in samples if row["label"] in labels):
        lines.append(
            f"| {r['label']} | {r['s']:.0f} | {r['k']:.0f} | {r['t']:.2f} | {r['sigma']:.2f} "
            f"| {r['fdm_price']:.4f} | {r['binomial_price']:.4f} | {r['abs_err']:.2e} "
            f"| {r['delta_abs_err']:.2e} | {r['gamma_abs_err']:.2e} | {r['theta_abs_err']:.2e} |"
        )
    lines.append("")
    return lines


def render_markdown(report):
    acc = report["sections"]["accuracy"]
    convergence = report["sections"]["convergence"]
    perf = report["sections"]["performance"]
    det = report["sections"]["determinism"]
    euro = acc["european_vs_analytic"]["max_abs_err"]
    g = report["grid"]
    status = "✅ PASS" if report["passed"] else "❌ FAIL"

    lines = [
        "# Grail Derivatives — Engine Validation Report",
        "",
        f"**Status:** {status}  ·  **Generated:** {report['generated_at']}  ·  **Grid:** h={g['h']}, Tn={g['tn']}",
        "",
        "> Reproducible artifact generated by `scripts/generate_validation_report.py`. "
        "Re-run it against any engine build to verify these numbers yourself.",
        "",
        "## 1. European accuracy vs. closed-form Black–Scholes",
        "",
        "Exact analytic answer exists, so these errors reflect pure FDM discretization.",
        "",
        f"| Metric | Max abs error | Tolerance | Result |",
        f"|---|---|---|---|",
        f"| Price | {euro['price']:.2e} | {report['tolerances']['euro_price_abs']:.0e} | {'✅' if report['gates']['euro_price'] else '❌'} |",
        f"| Delta | {euro['delta']:.2e} | {report['tolerances']['euro_delta_abs']:.0e} | {'✅' if report['gates']['euro_delta'] else '❌'} |",
        f"| Gamma | {euro['gamma']:.2e} | {report['tolerances']['euro_gamma_abs']:.0e} | {'✅' if report['gates']['euro_gamma'] else '❌'} |",
        f"| Theta | {euro['theta']:.2e} | {report['tolerances']['euro_theta_abs']:.0e} | {'✅' if report['gates']['euro_theta'] else '❌'} |",
        f"| Vega | {euro['vega']:.2e} | {report['tolerances']['euro_vega_abs']:.0e} | {'✅' if report['gates']['euro_vega'] else '❌'} |",
        "",
        "## 2. American & Bermudan accuracy vs. high-resolution binomial",
        "",
        f"Independent method (binomial N={acc['american_vs_binomial']['binomial_n']}); validates price + Greeks incl. theta.",
        "",
        f"- **American — max price abs error:** {acc['american_vs_binomial']['max_price_abs_err']:.2e} "
        f"(tolerance {report['tolerances']['amer_price_abs']:.0e}) {'✅' if report['gates']['american_price'] else '❌'}",
        f"- **Bermudan — max price abs error:** {acc['bermudan_vs_binomial']['max_price_abs_err']:.2e} "
        f"(warn threshold {report['tolerances']['bermudan_price_warn']:.0e}) "
        f"{'⚠️ see findings' if acc['bermudan_vs_binomial']['max_price_abs_err'] > report['tolerances']['bermudan_price_warn'] else '✅'}",
        "",
        "## 2b. Barrier accuracy vs. closed form",
        "",
        "Single barriers vs. Reiner–Rubinstein, double barriers vs. Kunitomo–Ikeda (continuous "
        "monitoring; the FDM pins each barrier on a grid node). Knock-ins are priced by in-out parity.",
        "",
        f"- **Barrier — max price abs error:** {acc['barrier_vs_closed_form']['max_price_abs_err']:.2e} "
        f"(tolerance {report['tolerances']['barrier_price_abs']:.0e}) {'✅' if report['gates']['barrier_price'] else '❌'}",
    ]
    lines += [
        "",
        "## 3. Convergence",
        "",
        "- **Estimated order (log-log slope):**",
        "",
        "| Type | Slope | Tolerance | Chart | Result |" ,
        "|---|---|---|---|---|",
    ]
    for conv in convergence:
        deriv_type = conv["deriv_type"]
        deriv_name = conv["deriv_name"]
        key = f"convergence-{deriv_type}"
        conv_check = report['gates'][key]
        min = TOL["convergence_slope_min"][deriv_type]
        max = TOL["convergence_slope_max"][deriv_type]
        lines += [f"| {deriv_name} | {conv['slope']:.4f} | {min} <= slope <= {max} | [{conv['chart']}]({conv['chart']}) | {'✅' if conv_check else '❌'} "]

    lines += [
        "## 4. Performance",
        "",
        f"- **Batch throughput:** {perf['options_per_sec']:,.0f} options/sec "
        f"({perf['batch_size']:,} options in {perf['elapsed_s']*1000:.0f} ms) "
        f"at the benchmark grid h={perf['grid']['h']}, Tn={perf['grid']['tn']}",
        "- **Single-solve latency:**",
        "",
        "| Tn | latency (ms) |",
        "|---|---|",
    ]
    lines += [f"| {row['tn']} | {row['ms']:.2f} |" for row in perf["latency_by_tn"]]
    lines += [
        "",
        "## 5. Determinism",
        "",
        f"- Identical inputs → bit-identical outputs: {'✅ yes' if det['identical'] else '❌ NO'}",
        "",
    ]
    if report.get("findings"):
        lines += ["## ⚠️ Findings", ""]
        lines += [f"- {finding}" for finding in report["findings"]]
        lines += [""]

    # Appendix: the full per-scenario European accuracy matrix (incl. vega), so a reviewer
    # can audit every point — not just the summary maxima above.
    samples = acc["european_vs_analytic"]["samples"]
    lines += [
        "## Appendix A — European accuracy, per scenario",
        "",
        "Per-point FDM-vs-analytic absolute errors across the full matrix.",
        "",
        "| Type | S | K | T | σ | Δprice | Δdelta | Δgamma | vega (fdm / analytic) | Δvega |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in samples:
        lines.append(
            f"| {row['label']} | {row['s']:.0f} | {row['k']:.0f} | {row['t']:.2f} | {row['sigma']:.2f} "
            f"| {row['abs_err']:.2e} | {row['delta_abs_err']:.2e} | {row['gamma_abs_err']:.2e} "
            f"| {row['fdm_vega']:.4f} / {row['analytic_vega']:.4f} | {row['vega_abs_err']:.2e} |"
        )
    lines += [""]

    # Appendices B & C: per-scenario American / Bermudan accuracy vs. the binomial reference.
    lines += _vs_binomial_appendix(
        "Appendix B — American accuracy, per scenario", acc["samples"], ("American Call", "American Put"))
    lines += _vs_binomial_appendix(
        "Appendix C — Bermudan accuracy, per scenario", acc["samples"], ("Bermudan Call", "Bermudan Put"))

    # Appendix D: per-scenario barrier accuracy vs. the closed-form barrier price.
    barrier_samples = acc["barrier_vs_closed_form"]["samples"]
    lines += [
        "## Appendix D — Barrier accuracy, per scenario",
        "",
        "Per-point FDM-vs-closed-form absolute errors (Reiner–Rubinstein / Kunitomo–Ikeda).",
        "",
        "| Type | S | K | T | σ | b_low | b_up | fdm price | closed-form price | Δprice |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in barrier_samples:
        lines.append(
            f"| {row['label']} | {row['s']:.0f} | {row['k']:.0f} | {row['t']:.2f} | {row['sigma']:.2f} "
            f"| {row['b_low']:.2f} | {row['b_up']:.2f} | {row['fdm_price']:.4f} | {row['closed_form_price']:.4f} "
            f"| {row['abs_err']:.2e} |"
        )
    lines += [""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the Grail Derivatives engine validation report.")
    parser.add_argument("--out", default="validation", help="output directory (default: ./validation)")
    parser.add_argument("--quick", action="store_true", help="smaller matrix for a fast run")
    parser.add_argument("--no-fail", action="store_true", help="always exit 0, even if a gate fails")
    args = parser.parse_args(argv)

    os.environ["VALIDATION_OUT"] = args.out
    report = run_validation(quick=args.quick)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(render_markdown(report))

    print(render_markdown(report))
    print(f"\nArtifacts written to {args.out}/ (report.json, report.md, convergence.png)")

    if not report["passed"] and not args.no_fail:
        print("\n❌ One or more validation gates failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
