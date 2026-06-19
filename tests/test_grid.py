import io

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pytest
from app.api import deriv_labels
from app.routers import pricing
from app.schemas.pricing import OptionConfigSchema

config = OptionConfigSchema(
    deriv=4,        # Bermudan Call
    Tn=100,         
    time=1.0,       
    h=1.0,          # Lock directly into the L1 hardware cache sweet spot
    r=0.1, 
    sigma=0.5, 
    s=100.0, 
    k=110.0,
    q=0.0,
    frequency=4
)
# 1. Create a lightweight dummy class to satisfy your UserClaims structure
class DummyUser:
    id = 1
    username = "quant_tester"
    role = "admin"

def plot_surface(price_surface, rows_Tn, cols_Xm, image_regression):
    if image_regression:
        matplotlib.use("Agg")

    # ============================================================
    # 📐 RECONSTRUCT MATRIX AXES COORDINATES (S and Time)
    # ============================================================
    # Time vector scales from 0.0 (Maturity) up to T = 1.0 year (Today)
    time_axis = np.linspace(0.0, config.time, rows_Tn)
    
    # Spatial asset axis: deduce boundaries from your FDM engine limits.
    # Since your linear 0.1 h-grid has 361 spatial nodes centered around S=100:
    # S_min is typically 0, S_max is derived inside your C++ library boundary rules (e.g., ~360)
    # Let's dynamically map it based on your column node count and h step thickness:
    S_min = 0.0
    S_max = (cols_Xm - 1) * config.h 
    asset_axis = np.linspace(S_min, S_max, cols_Xm)

    # Generate the 2D coordinate mesh required for Matplotlib 3D plotting
    X_time, Y_asset = np.meshgrid(time_axis, asset_axis)

    # Transpose the pricing surface matrix if your C++ memory layout is row-major 
    # to align perfectly with the (Time, Asset) coordinate mapping axes
    # (If your plot looks flipped 90 degrees, use price_surface instead)
    Z_price = price_surface.T

    # ============================================================
    # 🎨 DRAW THE INTERACTIVE 3D RISK SURFACE CHART
    # ============================================================
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
    
    # Use the 'viridis' or 'plasma' colormap to display premium depth clearly
    surface = ax.plot_surface(
        X_time, Y_asset, Z_price, 
        # pylint: disable=no-member
        cmap=cm.viridis, 
        linewidth=0, 
        antialiased=True,
        alpha=0.9
    )

    # Configure structural chart annotations and label matrices
    current_name = deriv_labels.get(config.deriv, "Unknown Option")
    ax.set_title(f"{current_name} FDM Pricing Surface $(\\tau, S)$", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Time-to-Maturity ($\\tau$ in Years)", fontsize=11, labelpad=10)
    ax.set_ylabel("Underlying Asset Spot Price ($S$)", fontsize=11, labelpad=10)
    ax.set_zlabel("Option Premium ($V$)", fontsize=11, labelpad=10)

    # Restrict view fields to your active trading sweet spot (e.g., S between 50 and 170)
    # to hide the trailing edge boundaries and magnify your strike kink detail!
    ax.set_ylim(S_min, S_max)

    # Add a smooth floating color legend bar index scale
    fig.colorbar(surface, shrink=0.5, aspect=10, pad=0.1, label="Premium Dollar Value ($)")

    # Enable native interactive pan/zoom rotations inside your workspace window frame
    ax.view_init(elev=25, azim=-125) 
    
    print("Displaying 3D option model chart...")
    if image_regression:
        img_buff = io.BytesIO()
        plt.savefig(img_buff, bbox_inches="tight")
        img_buff.seek(0)
        raw_image_bytes = img_buff.getvalue()
        image_regression.num_threshold = 0.01
        image_regression.check(raw_image_bytes)
    else:
        plt.show()        
    plt.close(fig)

@pytest.mark.asyncio
async def test_grid_endpoint(image_regression):
    response = await pricing.price_option_full_grid(config)
    raw_binary_payload = b"".join([chunk async for chunk in response.body_iterator]) if hasattr(response, 'body_iterator') else b"".join([chunk for chunk in response.iter_bytes()])
    
    if response.status_code == 200:
        # 1. Read your pricing Greeks straight out of the fast HTTP header metadata!
        print(f"Option Price: {response.headers.get('X-Price')}")
        print(f"Option Delta: {response.headers.get('X-Delta')}")
        
        # 2. Extract the exact array shape properties
        rows = int(response.headers.get("X-Grid-Rows-Tn"))
        cols = int(response.headers.get("X-Grid-Cols-Xm"))
        print(f"Rows Tn: {rows}")
        print(f"Cols Xm: {cols}")
        
        # 3. Instantly load the binary stream straight into a perfectly shaped 2D array!
        # Bypasses string conversions completely—finishes in microseconds.
        flat_data = np.frombuffer(raw_binary_payload, dtype=np.float64)
        
        print("\n================== 🔬 BINARY PAYLOAD GEOMETRY AUDIT ==================")
        print(f"Total 64-bit float elements received: {len(flat_data):,}")
        print(f"Expected elements from HTTP headers:  {rows} * {cols} = {rows * cols:,}")
        
        # 💥 THE DUMP GATE: If these numbers don't match, your size tracking is broken!
        if len(flat_data) != (rows * cols):
            print("🚨 CRITICAL ERROR: Transferred byte size does not match dimensions!")
        print("========================================================================")

        # Reconstruct the 2D surface
        price_surface_2d = flat_data.reshape((rows, cols))

        print("\n================== 📊 CONVERGED OPTIONS VALUE DUMP ==================")
        mid_space = int(cols / 2) # Roughly node 360 (Where S ≈ 100)
        
        print(f"Printing a 5x5 grid slice from the final, fully converged time rows:")
        # Look at the very end of your 1000 time steps (Rows 990 to 995!)
        for r in range(rows - 10, rows - 5): 
            row_str = " | ".join(f"{price_surface_2d[r, c]:8.4f}" for c in range(mid_space, mid_space + 5))
            print(f"Row [{r:03d}]: {row_str}")
        print("========================================================================")
        
        print(f"Successfully loaded pricing surface matrix! Shape: {price_surface_2d.shape}")
        
        plot_surface(price_surface_2d, rows, cols, image_regression)
    else:
        print(f"Filed loading endpoint status code {response.status_code}")
        
if __name__ == "__main__":
    import asyncio
    
    print("⏳ Launching async testing pipeline framework...")
    # ✅ THE ASYNC FIX: Feed the coroutine straight into the hardware event loop engine
    asyncio.run(test_grid_endpoint(False))
