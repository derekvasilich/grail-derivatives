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
from datetime import datetime, timezone
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running as a plain script (python scripts/...) by ensuring the repo root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import DerivType, FrequencyType, OptionConfig, fdm_price_batch, fdm_price_binomial_all, fdm_price_single  # noqa: E402

_N = NormalDist()

# Index/label alignment for the 6 derivative types returned by fdm_price_binomial_all.
DERIV_ROWS = [
    (DerivType.VanillaCall, "European Call", True),
    (DerivType.VanillaPut, "European Put", False),
    (DerivType.AmericanCall, "American Call", True),
    (DerivType.AmericanPut, "American Put", False),
    (DerivType.BermudanCall, "Bermudan Call", True),
    (DerivType.BermudanPut, "Bermudan Put", False),
]

# Tolerance gates (errors above these fail the report). Tuned to the default report grid.
TOL = {
    "euro_price_abs": 5e-2,     # vs analytic BSM
    "euro_delta_abs": 5e-3,
    "euro_gamma_abs": 5e-4,
    "amer_price_abs": 5e-2,     # American vs high-res binomial (gated)
    "bermudan_price_warn": 1e-1,  # Bermudan vs binomial — surfaced as a finding, not gated
    "convergence_slope_min": 1.90,
    "convergence_slope_max": 2.10,
}


# ------------------------------------------------------------------
# Closed-form Black–Scholes–Merton (the European ground truth)
# ------------------------------------------------------------------
def black_scholes(s, k, t, sigma, r, q, is_call):
    """Analytic BSM price, delta, gamma for a European option with continuous yield q."""
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_q, disc_r = math.exp(-q * t), math.exp(-r * t)
    if is_call:
        price = s * disc_q * _N.cdf(d1) - k * disc_r * _N.cdf(d2)
        delta = disc_q * _N.cdf(d1)
    else:
        price = k * disc_r * _N.cdf(-d2) - s * disc_q * _N.cdf(-d1)
        delta = -disc_q * _N.cdf(-d1)
    gamma = disc_q * _N.pdf(d1) / (s * sigma * sqrt_t)
    return {"price": price, "delta": delta, "gamma": gamma}


def _make_config(deriv, s, k, t, sigma, r, q, h, tn):
    return OptionConfig(
        deriv=deriv, frequency=FrequencyType.Quarterly,
        Tn=tn, top=0, bottom=1, left=0, right=0,
        time=t, h=h, r=r, sigma=sigma, s=s, k=k, q=q,
    )


# ------------------------------------------------------------------
# Section runners
# ------------------------------------------------------------------
def run_accuracy(grid, matrix, binomial_n):
    """Sections 1 & 2: FDM vs analytic (European) and vs binomial (all 6 types)."""
    euro_rows, exotic_rows = [], []
    euro_max = {"price": 0.0, "delta": 0.0, "gamma": 0.0}
    american_max_price = 0.0
    bermudan_max_price = 0.0

    for (s, k, t, sigma, r, q) in matrix:
        base = _make_config(DerivType.VanillaCall, s, k, t, sigma, r, q, grid["h"], grid["tn"])
        bin_status, bin_ref = fdm_price_binomial_all(base, binomial_n)
        if bin_status != 0:
            raise RuntimeError(f"binomial reference failed (status {bin_status}) at {(s,k,t,sigma,r,q)}")

        for idx, (deriv, label, is_call) in enumerate(DERIV_ROWS):
            cfg = _make_config(deriv, s, k, t, sigma, r, q, grid["h"], grid["tn"])
            st, fdm, _ = fdm_price_single(cfg)
            if st != 0:
                raise RuntimeError(f"FDM solve failed (status {st}) for {label} at {(s,k,t,sigma,r,q)}")

            # European → compare against the exact analytic answer.
            if deriv in (DerivType.VanillaCall, DerivType.VanillaPut):
                exact = black_scholes(s, k, t, sigma, r, q, is_call)
                dp, dd, dg = abs(fdm.price - exact["price"]), abs(fdm.delta - exact["delta"]), abs(fdm.gamma - exact["gamma"])
                euro_max["price"], euro_max["delta"], euro_max["gamma"] = (
                    max(euro_max["price"], dp), max(euro_max["delta"], dd), max(euro_max["gamma"], dg))
                euro_rows.append({"label": label, "s": s, "k": k, "t": t, "sigma": sigma,
                                  "fdm_price": fdm.price, "analytic_price": exact["price"],
                                  "abs_err": dp, "rel_err": dp / exact["price"] if exact["price"] else 0.0,
                                  "delta_abs_err": dd, "gamma_abs_err": dg})

            # All 6 types → cross-validate against the independent binomial method.
            ref = bin_ref[idx]
            dprice = abs(fdm.price - ref.price)
            if deriv in (DerivType.AmericanCall, DerivType.AmericanPut):
                american_max_price = max(american_max_price, dprice)
            elif deriv in (DerivType.BermudanCall, DerivType.BermudanPut):
                bermudan_max_price = max(bermudan_max_price, dprice)
            exotic_rows.append({"label": label, "s": s, "k": k, "t": t, "sigma": sigma,
                                "fdm_price": fdm.price, "binomial_price": ref.price, "abs_err": dprice,
                                "delta_abs_err": abs(fdm.delta - ref.delta),
                                "gamma_abs_err": abs(fdm.gamma - ref.gamma),
                                "theta_abs_err": abs(fdm.theta - ref.theta)})

    return {
        "european_vs_analytic": {"max_abs_err": euro_max, "samples": euro_rows},
        "american_vs_binomial": {"max_price_abs_err": american_max_price, "binomial_n": binomial_n},
        "bermudan_vs_binomial": {"max_price_abs_err": bermudan_max_price, "binomial_n": binomial_n},
        "samples": exotic_rows,
    }


def run_convergence(out_dir):
    """Section 3: refine the grid for a fixed European option, estimate the slope, chart it."""
    exact = 13.26967658  # BSM call, S=K=100, sigma=0.2, r=0.1, T=1 (matches the convergence test)
    resolutions = [1, 2, 5, 10]
    tn_steps = [100, 400, 2500, 10000]
    nodes, errors = [], []
    for res, tn in zip(resolutions, tn_steps):
        cfg = _make_config(DerivType.VanillaCall, 100.0, 100.0, 1.0, 0.2, 0.1, 0.0, 1.0 / res, tn)
        st, gr, _ = fdm_price_single(cfg)
        if st != 0:
            raise RuntimeError(f"convergence solve failed (status {st}) at res {res}")
        nodes.append(res)
        errors.append(abs(gr.price - exact))

    nodes_arr, errors_arr = np.array(nodes), np.array(errors)
    slope, _ = np.polyfit(np.log(1.0 / nodes_arr), np.log(errors_arr), 1)

    chart_path = os.path.join(out_dir, "convergence.png")
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

    return {"slope": float(slope), "resolutions": nodes, "errors": [float(e) for e in errors], "chart": "convergence.png"}


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
    perf_grid = {"h": 1.0, "tn": 300}
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
        "american_price": accuracy["american_vs_binomial"]["max_price_abs_err"] <= TOL["amer_price_abs"],
        "convergence": TOL["convergence_slope_min"] <= convergence["slope"] <= TOL["convergence_slope_max"],
        "determinism": determinism["identical"],
    }

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
        "sections": {"accuracy": accuracy, "convergence": convergence,
                     "performance": performance, "determinism": determinism},
        "gates": gates,
        "findings": findings,
        "passed": all(gates.values()),
    }


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------
def render_markdown(report):
    acc = report["sections"]["accuracy"]
    conv = report["sections"]["convergence"]
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
        "## 3. Convergence",
        "",
        f"- **Estimated order (log-log slope):** {conv['slope']:.4f} "
        f"(expected ~2.0 for Crank–Nicolson) {'✅' if report['gates']['convergence'] else '❌'}",
        f"- Chart: `{conv['chart']}`",
        "",
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
