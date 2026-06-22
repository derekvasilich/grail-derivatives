import pytest
import io
import numpy as np
import matplotlib.pyplot as plt

from app.api import OptionConfig, fdm_price_single, fdm_price_binomial_all
from scripts.generate_validation_report import DerivType, DERIV_ROWS, black_scholes

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
    deriv_type, deriv_name, is_call = deriv_row

    # Allocate a contiguous array of structs that Python can share with C
    results = [TaskResult() for _ in range(num_tasks)]
    for i, res in enumerate(resolutions):
        results[i].dim = int(res)
        results[i].rmse = 0.0
        results[i].price = 0.0

    config = OptionConfig(
        deriv=deriv_type,
        left=0, 
        right=0,
        time=1.0, 
        r=0.1, 
        sigma=0.2, 
        s=100.0, 
        k=110.0,
        q=0.0,
    )

    if deriv_type == DerivType.VanillaCall or deriv_type == DerivType.VanillaPut:
        exact = black_scholes(config.s, config.k, config.time, config.sigma, config.r, config.q, is_call)
    else:
        success, all_greeks = fdm_price_binomial_all(config, 8000)
        greeks = all_greeks[deriv_type]
        exact = {
            "price": greeks.price,
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "theta": greeks.theta
        }
        
    print(f"\n   Tn,    H,  PRICE,  EXACT,   RMSE")
    # 1. Initialize the input configuration values
    for i, res in enumerate(resolutions):
        local_conf = OptionConfig.from_buffer_copy(config)
        local_conf.h = 1/res
        local_conf.Tn = tn_steps[i]
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
    # (Allowing a normal numerical variance window between 1.8 and 2.2)
    if deriv_type == DerivType.BermudanPut:
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