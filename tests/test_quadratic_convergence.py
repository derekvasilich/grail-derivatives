import pytest
import io
import numpy as np
import matplotlib.pyplot as plt

from app.api import fdm_price_single, fdm_price_binomial_all
from scripts.generate_validation_report import (
    DerivType, DERIV_ROWS, ORACLE_BINOMIAL, config_for_row, reference_price)

class TaskResult:
    def __init__(self):
        self.dim = 0
        self.rmse = 0.0
        self.price = 0.0

@pytest.mark.parametrize("deriv_row", DERIV_ROWS)
def test_convergence(image_regression, deriv_row):
    # Define grid configurations to test
    resolutions = [1, 2, 5, 10]
    tn_steps = [100, 400, 2500, 10000]
    num_tasks = len(resolutions)
    row = deriv_row
    deriv_type, deriv_name, is_call = row.deriv, row.label, row.is_call

    # Allocate a contiguous array of structs that Python can share with C
    results = [TaskResult() for _ in range(num_tasks)]
    for i, res in enumerate(resolutions):
        results[i].dim = int(res)
        results[i].rmse = 0.0
        results[i].price = 0.0

    # Fixed market point for the refinement study. Each row is validated against its own
    # independent oracle: Europeans -> analytic, American/Bermudan -> binomial, barriers ->
    # the closed-form barrier price (so the slope measures true error decay, never FDM-vs-FDM).
    s, k, t, sigma, r, q = 100.0, 110.0, 1.0, 0.2, 0.1, 0.0

    # Helper: this test historically ran Bermudans with frequency=0 (no quarterly schedule),
    # which fixes the binomial reference convention it converges against. Preserve that here so
    # the slope window stays calibrated; barriers ignore frequency entirely.
    def _conf(h, tn):
        c = config_for_row(row, s, k, t, sigma, r, q, h, tn)
        c.frequency = 0
        return c

    bin_ref = None
    if row.oracle == ORACLE_BINOMIAL:
        _, bin_ref = fdm_price_binomial_all(_conf(1.0, 100), 8000)
    exact = reference_price(row, s, k, t, sigma, r, q, bin_ref)

    print(f"\n   Tn,    H,  PRICE,  EXACT,   RMSE")
    # 1. Initialize the input configuration values
    for i, res in enumerate(resolutions):
        local_conf = _conf(1.0 / res, tn_steps[i])
        status, greeks, prices = fdm_price_single(local_conf)

        assert status == 0

        results[i].price = greeks.price
        results[i].rmse = np.fabs(greeks.price - exact['price'])
        print(f"{local_conf.Tn:5d}, {local_conf.h:.2f}, {greeks.price:.4f}, {exact['price']:.4f}, {results[i].rmse:.4f}")

    # 4. Call the method using byref()

    # # Extract results back into native NumPy arrays
    nodes = np.array([task.dim for task in results])
    errors = np.array([task.rmse for task in results])

    # # Calculate Log-Log linear regression slope
    X_log = np.log(1/nodes)
    Y_log = np.log(errors)
    slope, _ = np.polyfit(X_log, Y_log, 1)

    print("=" * 60)
    print(f"   ⭐ {deriv_name} CONVERGENCE SLOPE VALUE: {slope:.4f} ⭐")
    print("=" * 60)

    # 5. Assert that your Crank-Nicolson/Implicit engine achieves 2nd order accuracy
    # (Allowing a normal numerical variance window). Barriers sit exactly on a grid node and
    # recover clean 2nd order, but the strike+barrier interaction makes the slope a touch noisier,
    # so they get the same relaxed window as the Bermudan early-exercise row.
    is_barrier = row.oracle in ("barrier", "dbl_barrier")
    if deriv_type == DerivType.BermudanPut or is_barrier:
        assert 1.75 <= slope <= 2.25, f"FDM grid spatial convergence failed. Slope: {slope}"
    else:
        assert 1.95 <= slope <= 2.05, f"FDM grid spatial convergence failed. Slope: {slope}"

    # # # Generate chart visualization
    plt.figure(figsize=(8, 6))
    plt.loglog(nodes, errors, 'o-', color='crimson', linewidth=2, label=f'Crank-Nicolson (Slope: {slope:.4f})')
    plt.loglog(nodes, (errors[0] * (nodes[0]/nodes)**2), '--', color='dodgerblue', label='Theoretical O(Δx²) Slope')
    plt.title(f'Log-Log Convergence Analysis: {deriv_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Grid Resolution Node Count (Xm / Tn)')
    plt.ylabel('Unskewed Interior RMSE')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    img_buff = io.BytesIO()
    plt.savefig(img_buff, bbox_inches="tight")
    img_buff.seek(0)
    raw_image_bytes = img_buff.getvalue()
    if image_regression:
        image_regression.num_threshold = 0.01
        image_regression.check(raw_image_bytes)
    plt.close()

if __name__ == "__main__":
    [test_convergence(False, k) for k in DERIV_ROWS]