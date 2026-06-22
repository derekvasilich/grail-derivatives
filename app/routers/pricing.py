import ctypes
from io import BytesIO
from typing import List
import structlog
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query, Request, Response, status as http_status
from app.api import OptionConfig, fdm_price_batch, fdm_price_single
from app.schemas.pricing import CompactGreeksResponse, OptionConfigSchema

router = APIRouter()

# ------------------------------------------------------------------
# Reusable OpenAPI response documentation shared across pricing routes.
# FastAPI merges these into the generated Swagger / ReDoc schema so every
# authenticated endpoint advertises the same auth + engine failure modes.
# ------------------------------------------------------------------
AUTH_RESPONSES: dict = {
    401: {
        "description": "**Unauthorized** — the request carried no bearer token, or the "
                       "JWT failed asymmetric signature validation against the Cognito JWKS.",
    },
    403: {
        "description": "**License Expired** — the proprietary compiled C++ pricing core "
                       "rejected the call (engine status `-99`). Renew the engine license.",
    },
    422: {
        "description": "**Validation Error** — one or more `OptionConfig` fields failed "
                       "Pydantic schema validation (wrong type, missing `deriv`, etc.).",
    },
    500: {
        "description": "**Engine Failure** — the C++ numerical solver returned a non-zero "
                       "status mid-calculation (matrix instability or memory fault).",
    },
}

VEGA_QUERY = Query(
    False,
    description=(
        "When `true`, run the extra high-precision volatility **bump-and-reprice** passes "
        "to populate `vega`. Left `false` by default because the additional re-solve roughly "
        "doubles compute cost; `vega` is returned as `0.0` when skipped."
    ),
)


@router.post(
    "/pricing/batch",
    response_model=List[CompactGreeksResponse],
    tags=["Pricing"],
    summary="Price a batch of options concurrently (JSON)",
    response_description="A parallel array of compact Greeks — one entry per submitted config, in request order.",
    responses={
        400: {"description": "**Empty Batch** — the submitted JSON array contained zero option configs."},
        **AUTH_RESPONSES,
    },
)
async def price_options_batch(
    payload: List[OptionConfigSchema],
    calculate_vega: bool = VEGA_QUERY,
):
    """
    ⚡ **High-Throughput Sweep Pipe**

    Prices an entire **array** of option contracts in a single round-trip, fanning the
    workload across the OpenMP multi-core C++ loops using dense contiguous vector streams.
    Ideal for risk-desk scenario sweeps and overnight revaluation jobs.

    **Request body**: a JSON list of `OptionConfig` objects (see the schema below).

    **Returns**: a `CompactGreeksResponse` list of identical length, index-aligned to the
    input — element `i` of the response is the valuation of element `i` of the request.

    **Throughput**: ~17,500+ options/sec on a typical multi-core node. For the absolute
    lowest latency on very large books, prefer the zero-copy `POST /v1/pricing/binary` pipe.
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
    status, greeks_output_vector = fdm_price_batch(configs_vector, batch_size, vega_flag)
    if status == -99:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="The underlying proprietary quantitative processing core license has expired."
        )
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

@router.post(
    "/pricing/binary",
    tags=["Pricing"],
    summary="Zero-copy binary pricing pipe (octet-stream in/out)",
    response_description="Raw octet-stream of packed Greeks structs — one per input config, in order.",
    responses={
        200: {
            "description": "Binary buffer of computed Greeks structs, index-aligned to the input configs.",
            "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
        },
        400: {
            "description": "**Misaligned Payload** — the body length is zero or not an exact "
                           "multiple of `sizeof(OptionConfig)`, so the struct array cannot be reconstructed.",
        },
        **AUTH_RESPONSES,
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                        "description": "Contiguous little-endian array of packed C `OptionConfig` structs. "
                                       "Total byte length MUST be an exact multiple of the struct size.",
                    }
                }
            },
        }
    },
)
async def price_options_binary(request: Request):
    """
    ⚡ **Institutional High-Throughput Pipeline**

    The fastest ingress path: accepts a raw, contiguous binary array of native C
    `OptionConfig` structs as the POST body and pins a `ctypes` pointer **directly** onto
    the socket buffer — bypassing JSON parsing entirely for absolute zero-copy execution.

    **Request body** (`application/octet-stream`): tightly packed `OptionConfig` structs.
    The total length must be an exact multiple of `sizeof(OptionConfig)` or the request is
    rejected with `400`.

    **Returns** (`application/octet-stream`): the raw Greeks struct buffer, one record per
    input config, in the same order — ready to be `memcpy`'d straight back into a client array.

    > ⚠️ This pipe trades JSON ergonomics for raw speed. Clients are responsible for matching
    > the exact struct memory layout and endianness of the compiled engine.
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
    if status == -99:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="The underlying proprietary quantitative processing core license has expired."
        )
    if status != 0:
        raise HTTPException(status_code=500, detail="C++ pricing matrix calculation crashed.")

    # 6. Stream the raw calculation output bytes straight back down the network pipe!
    return Response(
        content=bytes(c_greeks_buffer), 
        media_type="application/octet-stream"
    )


# We remove the response_model=GridPricingResponse constraint to prevent JSON conversion
@router.post(
    "/pricing/grid",
    tags=["Pricing"],
    summary="Compute the full 2D pricing surface (binary float64)",
    response_description="Raw row-major float64 surface buffer; scalar Greeks and grid geometry are returned in X-* headers.",
    responses={
        200: {
            "description": "Row-major `float64` array of the full `Tn × Xm` price surface. "
                           "Scalar risk metrics travel out-of-band in custom response headers to keep the payload clean.",
            "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
            "headers": {
                "X-Price": {"description": "Option premium at spot, 8dp.", "schema": {"type": "string"}},
                "X-Delta": {"description": "∂V/∂S, 8dp.", "schema": {"type": "string"}},
                "X-Gamma": {"description": "∂²V/∂S², 8dp.", "schema": {"type": "string"}},
                "X-Theta": {"description": "∂V/∂t, 8dp.", "schema": {"type": "string"}},
                "X-Vega": {"description": "∂V/∂σ, 8dp.", "schema": {"type": "string"}},
                "X-Grid-Rows-Tn": {"description": "Number of temporal rows (Tn) in the surface.", "schema": {"type": "integer"}},
                "X-Grid-Cols-Xm": {"description": "Number of spatial columns (Xm) in the surface.", "schema": {"type": "integer"}},
            },
        },
        **AUTH_RESPONSES,
    },
)
async def price_option_full_grid(
    config: OptionConfigSchema,
):
    """
    🌳 **Quant Grid Engine**

    Computes the **entire** 2D asset-time pricing surface in one solver pass and streams it
    back as a raw `float64` binary buffer — deliberately avoiding JSON serialization of a
    large matrix, which would otherwise freeze browsers and bloat transfer size.

    **Request body**: a single `OptionConfig` object.

    **Returns** (`application/octet-stream`): the surface as a flat, **row-major**
    `Tn × Xm` array of 64-bit floats. Reconstruct it with the dimensions from the headers,
    e.g. `numpy.frombuffer(body, dtype='<f8').reshape((Tn, Xm))`.

    **Headers**: the scalar Greeks (`X-Price`, `X-Delta`, `X-Gamma`, `X-Theta`, `X-Vega`)
    and grid geometry (`X-Grid-Rows-Tn`, `X-Grid-Cols-Xm`) ride in custom HTTP headers so
    the binary body stays a pure numeric payload.
    """
    c_config = OptionConfig()
    for field, _ in config.model_fields.items():
        setattr(c_config, field, getattr(config, field))

    status, c_greeks, c_prices_surface_ptr = fdm_price_single(c_config)
    if status == -99:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="The underlying proprietary quantitative processing core license has expired."
        )
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
        # X-SPACE: the engine solves on a uniform log-price grid. Columns are at
        # S_m = exp(x_min + m*dx) (geometric in S). These let the client rebuild the S axis.
        "X-Log-Xmin": f"{c_greeks.x_min:.10f}",
        "X-Log-Dx": f"{c_greeks.dx:.10f}",
        "Content-Disposition": "attachment; filename=option_surface.bin"
    }

    # 3. Stream the raw continuous bytes directly down the pipe!
    binary_stream = BytesIO(surface_matrix.tobytes())
    return StreamingResponse(binary_stream, media_type="application/octet-stream", headers=headers)


@router.post(
    "/pricing/single",
    response_model=CompactGreeksResponse,
    tags=["Pricing"],
    summary="Price a single option and return its Greeks",
    response_description="Compact Greeks block (price, delta, gamma, theta, vega) plus the resolved grid dimensions (Tn × Xm).",
    responses={**AUTH_RESPONSES},
)
async def price_option_single(
    config: OptionConfigSchema,
    calculate_vega: bool = VEGA_QUERY,
):
    """
    🎯 **Single Target Engine**

    Prices one option scenario with a single tight solver pass and returns **only** the
    compact Greeks — no dense surface matrix overhead. This is the everyday endpoint for
    quoting, what-if analysis, and low-latency single-contract valuation.

    **Request body**: a single `OptionConfig` object.

    **Returns**: a `CompactGreeksResponse` (`price`, `delta`, `gamma`, `theta`, `vega`)
    plus the grid geometry (`Tn`, `Xm`) the engine resolved for the solve.

    Need the full price surface or a chart instead? See `POST /v1/pricing/grid` and
    `POST /v1/pricing/chart`.
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
    if status == -99:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="The underlying proprietary quantitative processing core license has expired."
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
    
@router.post(
    "/pricing/chart",
    tags=["Pricing"],
    summary="Render the pricing surface as a 3D PNG chart",
    response_description="A pre-rendered high-resolution PNG image of the 3D pricing surface.",
    responses={
        200: {
            "description": "A `150 DPI` Matplotlib 3D surface plot of the option premium over the (τ, S) mesh.",
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
        },
        **AUTH_RESPONSES,
    },
)
async def generate_pricing_surface_chart(
    config: OptionConfigSchema,
    calculate_vega: bool = VEGA_QUERY,
):
    """
    📊 **On-Demand Visualization Engine**

    Computes the full 2D pricing grid and returns a **pre-rendered** high-resolution 3D
    surface plot as a PNG — moving the Matplotlib rendering cost server-side so thin
    front-ends can display the surface without shipping or plotting the raw matrix.

    **Request body**: a single `OptionConfig` object.

    **Returns** (`image/png`): a 150 DPI `plasma`-colormap surface of option premium `V`
    over the time-to-maturity (`τ`) and underlying spot (`S`) axes, titled by derivative type.
    """
    # 1. Reuse your working internal single-option C++ FDM solver pass
    c_config = OptionConfig()
    for field, _ in config.model_fields.items():
        setattr(c_config, field, getattr(config, field))
        
    vega_flag = 1 if calculate_vega else 0

    status, c_greeks, c_prices_surface_ptr = fdm_price_single(c_config, vega_flag)
    if status == -99:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="The underlying proprietary quantitative processing core license has expired."
        )
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

    # 3. Build coordinate meshgrid tracking geometry.
    # X-SPACE: the engine solves on a uniform log-price grid, so the asset axis is
    # geometric: S_m = exp(x_min + m*dx). Reconstruct the true S coordinates per column.
    time_axis = np.linspace(0.0, config.time, rows_Tn)
    asset_axis = np.exp(c_greeks.x_min + np.arange(cols_Xm) * c_greeks.dx)
    S_min = float(asset_axis[0])
    S_max = float(asset_axis[-1])
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