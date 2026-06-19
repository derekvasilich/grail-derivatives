import ctypes
from io import BytesIO
from typing import List
import structlog
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# ✅ CRITICAL: Force Matplotlib to use a headless background backend 
# This prevents the server from trying to open an interactive GUI window on your AWS instances!
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from app.auth.jwt import UserClaims, get_current_user
from app.black_api import OptionConfig, fdm_price_batch, fdm_price_single
from app.schemas.pricing import CompactGreeksResponse, OptionConfigSchema

router = APIRouter()

@router.post("/pricing/batch", response_model=List[CompactGreeksResponse])
async def price_options_batch(
    payload: List[OptionConfigSchema], 
    calculate_vega: bool = Query(False, description="Compute high-precision volatility bump-and-scale passes"),
    user: UserClaims = Depends(get_current_user),
):
    """
    ⚡ HIGH-THROUGHPUT SWEEP PIPE: Prices thousands of options concurrently 
    over OpenMP multi-core processor threads using dense contiguous vector streams.
    """
    batch_size = len(payload)
    if batch_size == 0:
        raise HTTPException(status_code=400, detail="Payload batch array cannot be empty.")

    # Allocate clean sequential unmanaged memory segments via type multiplication
    configs_array_type = OptionConfig * batch_size    
    configs_vector = configs_array_type()

    # Populate arrays seamlessly using Pydantic field lookups
    for i, item in enumerate(payload):
        for field, _ in item.model_fields.items():
            setattr(configs_vector[i], field, getattr(item, field))

    # Convert the boolean parameter flag directly into a flat C-style integer value
    vega_flag = 1 if calculate_vega else 0

    # Execute the C++ core processing array block instantly
    status, greeks_output_vector = fdm_price_batch(configs_vector, vega_flag)
    if status != 0:
        raise HTTPException(status_code=500, detail="C++ core loop process failed during batch evaluation.")

    # Pack the flat data layout into your structured response schema arrays
    response = [
        CompactGreeksResponse(
            price=greeks_output_vector[i].price,
            delta=greeks_output_vector[i].delta,
            gamma=greeks_output_vector[i].gamma,
            theta=greeks_output_vector[i].theta,
            vega=greeks_output_vector[i].vega,
            Tn=greeks_output_vector[i].Tn,
            Xm=greeks_output_vector[i].Xm,
        ) for i in range(batch_size)
    ]
    return response

@router.post("/pricing/binary")
async def price_options_binary(request: Request):
    """
    ⚡ INSTUTIONAL HIGH-THROUGHPUT PIPELINE:
    Accepts a raw contiguous binary array of OptionConfig structs via a POST body.
    Bypasses JSON completely, achieving absolute zero-copy execution speed.
    """
    # 1. Stream the raw byte chunk straight out of the network socket memory buffer
    raw_body_bytes = await request.body()
    
    # 2. Extract structural boundaries natively
    struct_size = ctypes.sizeof(OptionConfig)
    total_bytes = len(raw_body_bytes)
    
    if total_bytes == 0 or total_bytes % struct_size != 0:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid binary payload alignment. Size must be a multiple of {struct_size} bytes."
        )
        
    batch_size = total_bytes // struct_size

    # 3. ✅ THE TRUE ZERO-COPY HOOK:
    # Pin a ctypes pointer instance directly onto the raw socket bytes buffer channel!
    configs_pointer = ctypes.cast(raw_body_bytes, ctypes.POINTER(OptionConfig))

    # 4. Fire your ultra-optimized 17,500+ options/sec C++ parallel loops
    status, c_greeks_buffer = fdm_price_batch(
        configs_pointer,
        batch_size,
        False # Default vega tracking off for speed
    )

    if status != 0:
        raise HTTPException(status_code=500, detail="C++ pricing matrix calculation crashed.")

    # 6. Stream the raw calculation output bytes straight back down the network pipe!
    return Response(
        content=bytes(c_greeks_buffer), 
        media_type="application/octet-stream"
    )


# We remove the response_model=GridPricingResponse constraint to prevent JSON conversion
@router.post("/pricing/grid")
async def price_option_full_grid(
    config: OptionConfigSchema,
    user: UserClaims = Depends(get_current_user),
):
    """
    🌳 QUANT GRID ENGINE: Computes the entire 2D asset-time surface
    and streams it back instantly as a raw binary byte array (float64) 
    to prevent browser freezing and maximize transfer speeds.
    """
    c_config = OptionConfig()
    for field, _ in config.model_fields.items():
        setattr(c_config, field, getattr(config, field))

    status, c_greeks, c_prices_surface_ptr = fdm_price_single(c_config)
    if status != 0:
        raise HTTPException(status_code=500, detail="C++ pricing matrix calculation crashed.")

    # Calculate exact array geometry bounds
    total_elements = c_greeks.Tn * c_greeks.Xm

     # 3. ✅ THE SAFE UNPACKING: Pull the raw integer address value directly!
    raw_address = c_prices_surface_ptr.value

    # Security Gate: Verify the address is no longer NULL (0x0)
    if not raw_address or raw_address == 0:
        raise HTTPException(status_code=500, detail="C++ failed to update the matrix memory address.")

    # Cast the raw address integer straight into your continuous 1D double block segment
    raw_buffer = ctypes.cast(raw_address, ctypes.POINTER(ctypes.c_double * total_elements))
    
    # Extract your 64-bit float bytes cleanly into Python isolates
    surface_matrix = np.frombuffer(raw_buffer.contents, dtype=np.float64, count=total_elements).copy()

    # 2. Pack metadata (Greeks + Matrix Dimensions) directly into custom HTTP Headers
    # This keeps your binary payload clean while still delivering your calculated risk metrics!
    headers = {
        "X-Price": f"{c_greeks.price:.8f}",
        "X-Delta": f"{c_greeks.delta:.8f}",
        "X-Gamma": f"{c_greeks.gamma:.8f}",
        "X-Theta": f"{c_greeks.theta:.8f}",
        "X-Vega": f"{c_greeks.vega:.8f}",
        "X-Grid-Rows-Tn": str(c_greeks.Tn),
        "X-Grid-Cols-Xm": str(c_greeks.Xm),
        "Content-Disposition": "attachment; filename=option_surface.bin"
    }

    # 3. Stream the raw continuous bytes directly down the pipe!
    binary_stream = BytesIO(surface_matrix.tobytes())
    return StreamingResponse(binary_stream, media_type="application/octet-stream", headers=headers)


@router.post("/pricing/single", response_model=CompactGreeksResponse)
async def price_option_single(
    config: OptionConfigSchema,
    calculate_vega: bool = Query(False, description="Compute high-precision volatility bump-and-scale passes"),
    user: UserClaims = Depends(get_current_user),
):
    """
    🎯 SINGLE TARGET ENGINE: Prices a single option scenario instantly, 
    returning only the final compact Greeks layout with zero surface-grid data overhead.
    """
    c_config = OptionConfig()
    for field, _ in config.model_fields.items():
        setattr(c_config, field, getattr(config, field))

    vega_flag = 1 if calculate_vega else 0

    # Fire your single options solver pass
    status, c_greeks, c_prices_surface_ptr = fdm_price_single(
        c_config,
        vega_flag
    )
    if status != 0:
        raise HTTPException(status_code=500, detail="C++ single option pricing calculation crashed.")

    return CompactGreeksResponse(
        price=c_greeks.price, 
        delta=c_greeks.delta, 
        gamma=c_greeks.gamma, 
        theta=c_greeks.theta, 
        vega=c_greeks.vega,
        Tn=c_greeks.Tn,
        Xm=c_greeks.Xm,
    )
    
@router.post("/pricing/chart")
async def generate_pricing_surface_chart(
    config: OptionConfigSchema,
    calculate_vega: bool = Query(False, description="Compute high-precision volatility passes"),
    user: UserClaims = Depends(get_current_user),
):
    """
    📊 ON-DEMAND VISUALIZATION ENGINE: Computes the full 2D pricing grid 
    and returns a pre-rendered, high-resolution 3D surface chart as a raw PNG image.
    """
    # 1. Reuse your working internal single-option C++ FDM solver pass
    c_config = OptionConfig()
    for field, _ in config.model_fields.items():
        setattr(c_config, field, getattr(config, field))
        
    vega_flag = 1 if calculate_vega else 0

    status, c_greeks, c_prices_surface_ptr = fdm_price_single(c_config, vega_flag)
    if status != 0:
        raise HTTPException(status_code=500, detail="C++ solver matrix calculation crashed.")

    # 2. Extract dimensions and unpack the raw unmanaged memory safely
    rows_Tn = c_greeks.Tn
    cols_Xm = c_greeks.Xm
    total_elements = rows_Tn * cols_Xm

    raw_address = c_prices_surface_ptr.contents if hasattr(c_prices_surface_ptr, 'contents') else c_prices_surface_ptr
    raw_buffer = ctypes.cast(raw_address, ctypes.POINTER(ctypes.c_double * total_elements))
    surface_matrix = np.frombuffer(raw_buffer.contents, dtype=np.float64, count=total_elements).copy()
    
    # Reshape matching your true Row-Major layout
    price_surface_2d = surface_matrix.reshape((rows_Tn, cols_Xm))

    # 3. Build coordinate meshgrid tracking geometry
    time_axis = np.linspace(0.0, config.time, rows_Tn)
    S_min = 0.0
    S_max = (cols_Xm - 1) * config.h
    asset_axis = np.linspace(S_min, S_max, cols_Xm)
    X_time, Y_asset = np.meshgrid(time_axis, asset_axis)
    Z_price = price_surface_2d.T  # Transpose to mate perfectly with meshgrid shapes

    # 4. Generate the Matplotlib 3D Figure
    matplotlib.use("Agg") 
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7), dpi=150)
    surface = ax.plot_surface(X_time, Y_asset, Z_price, cmap="plasma", linewidth=0, antialiased=True, alpha=0.9)
    
    # Dynamic title string assignment mapping
    deriv_labels = {0: "Vanilla Call", 1: "Vanilla Put", 2: "American Call", 3: "American Put", 4: "Bermudan Call", 5: "Bermudan Put"}
    option_title = deriv_labels.get(config.deriv, "Option")
    ax.set_title(f"{option_title} FDM Pricing Surface $(\\tau, S)$", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Time-to-Maturity ($\\tau$ in Years)", fontsize=9, labelpad=8)
    ax.set_ylabel("Underlying Asset Spot Price ($S$)", fontsize=9, labelpad=8)
    ax.set_zlabel("Option Premium ($V$)", fontsize=9, labelpad=8)
    
    # Crop the viewport sweet spot dynamically to trim edge singularities out of view
    # ax.set_ylim(config.s * 0.4, config.s * 1.8)
    ax.set_ylim(S_min, S_max)
    fig.colorbar(surface, shrink=0.5, aspect=12, pad=0.1, label="Premium Value ($)")
    ax.view_init(elev=22, azim=-128)

    # 5. ✅ THE BINARY FLUSH: Save the plot into an in-memory memory byte buffer
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    plt.close(fig) # Prevent memory leaks by closing the figure context explicitly
    img_buf.seek(0)

    # 6. Stream the raw image bytes straight down the HTTP socket pipe
    return Response(content=img_buf.getvalue(), media_type="image/png")