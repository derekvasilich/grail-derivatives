import ctypes
import io
import numpy as np
import matplotlib.pyplot as plt

from app.black_api import DerivType, OptionConfig, fdm_price_single

black_scholes_exact = 13.26967658

class TaskResult:
    def __init__(self):
        self.dim = 0
        self.rmse = 0.0
        self.price = 0.0

def test_convergence(image_regression):
    # Define grid configurations to test
    resolutions = [1, 2, 5, 10]
    tn_steps = [100, 400, 2500, 10000]
    num_tasks = len(resolutions)

    # Allocate a contiguous array of structs that Python can share with C
    results = [TaskResult() for _ in range(num_tasks)]
    for i, res in enumerate(resolutions):
        results[i].dim = int(res)
        results[i].rmse = 0.0
        results[i].price = 0.0

    # 1. Initialize the input configuration values
    for i, res in enumerate(resolutions):
        config = OptionConfig(
            deriv=DerivType.VanillaCall,
            Tn=tn_steps[i],
            left=0, 
            right=0,
            time=1.0, 
            h=1/res, 
            r=0.1, 
            sigma=0.2, 
            s=100.0, 
            k=100.0
        )
        status, greeks, prices = fdm_price_single(config)
        
        assert status == 0
            
        results[i].price = greeks.price
        results[i].rmse = np.fabs(greeks.price - black_scholes_exact)

    # 4. Call the method using byref()

    # # Extract results back into native NumPy arrays
    nodes = np.array([task.dim for task in results])
    errors = np.array([task.rmse for task in results])

    # # Calculate Log-Log linear regression slope
    X_log = np.log(1/nodes)
    Y_log = np.log(errors)
    slope, _ = np.polyfit(X_log, Y_log, 1)

    print(results)

    print("=" * 60)
    print(f"   ⭐ GLOBAL CONVERGENCE SLOPE VALUE: {slope:.4f} ⭐")
    print("=" * 60)

    # 5. Assert that your Crank-Nicolson/Implicit engine achieves 2nd order accuracy
    # (Allowing a normal numerical variance window between 1.8 and 2.2)
    assert 1.95 <= slope <= 2.05, f"FDM grid spatial convergence failed. Slope: {slope}"

    # # # Generate chart visualization
    plt.figure(figsize=(8, 6))
    plt.loglog(nodes, errors, 'o-', color='crimson', linewidth=2, label=f'Crank-Nicolson (Slope: {slope:.4f})')
    plt.loglog(nodes, (errors[0] * (nodes[0]/nodes)**2), '--', color='dodgerblue', label='Theoretical O(Δx²) Slope')
    plt.title('Log-Log Convergence Analysis: OpenMP Batch Engine', fontsize=12, fontweight='bold')
    plt.xlabel('Grid Resolution Node Count (Xm / Tn)')
    plt.ylabel('Unskewed Interior RMSE')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    if image_regression:
        img_buff = io.BytesIO()
        plt.savefig(img_buff, bbox_inches="tight")
        img_buff.seek(0)
        raw_image_bytes = img_buff.getvalue()
        image_regression.num_threshold = 0.01
        image_regression.check(raw_image_bytes)
    else:
        plt.show()

if __name__ == "__main__":
    test_convergence(False)