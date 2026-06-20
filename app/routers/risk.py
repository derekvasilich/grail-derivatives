"""Portfolio risk-scenario engine.

Shocks one or two market factors across a grid, re-prices the whole book at every
node via a SINGLE batch call into the C++ engine, and aggregates portfolio P&L and
net Greeks per node in pure Python. No engine changes required — the scenario grid is
just expanded into a contiguous OptionConfig array and handed to fdm_price_batch.
"""
import itertools

from fastapi import APIRouter, HTTPException, status as http_status

from app.api import OptionConfig, fdm_price_batch
from app.routers.pricing import AUTH_RESPONSES
from app.schemas.pricing import OptionConfigSchema
from app.schemas.risk import (
    FACTOR_FIELD,
    ScenarioNode,
    ScenarioRequest,
    ScenarioResponse,
    ScenarioSummary,
)

router = APIRouter()

# Hard ceiling on total valuations (positions × nodes + base block). Sub-second at this
# size; larger books/grids should move to an async job (see roadmap).
MAX_SCENARIO_VALUATIONS = 50_000

# Lower clamps so an extreme shock can't hand the solver a non-physical input.
_TIME_FLOOR = 1e-6
_POSITIVE_FLOOR = 1e-9

# Iterate field names off the class (avoids the deprecated instance-level model_fields).
_CONFIG_FIELDS = tuple(OptionConfigSchema.model_fields.keys())


def _fill_config(c_config: OptionConfig, base: OptionConfigSchema, overrides: dict) -> None:
    """Copy a position's base config into a ctypes struct, then apply factor overrides."""
    for field in _CONFIG_FIELDS:
        setattr(c_config, field, getattr(base, field))
    for field, value in overrides.items():
        setattr(c_config, field, value)


def _shocked_value(field: str, base_value: float, mode: str, shock: float) -> float:
    new_value = base_value * (1.0 + shock) if mode == "relative" else base_value + shock
    if field == "time":
        return max(new_value, _TIME_FLOOR)
    if field in ("s", "sigma"):
        return max(new_value, _POSITIVE_FLOOR)
    return new_value


@router.post(
    "/risk/scenario",
    response_model=ScenarioResponse,
    tags=["Risk"],
    summary="Stress a portfolio across a market-factor grid",
    response_description="Per-node portfolio P&L and net Greeks, plus a worst-case summary.",
    responses={
        400: {"description": "**Empty portfolio** or otherwise unrunnable scenario."},
        413: {"description": "**Scenario too large** — positions × nodes exceeds the valuation cap."},
        422: {"description": "**Invalid scenario** — bad shock spec, duplicate factor, or mixed underlyings under a spot shock."},
        **AUTH_RESPONSES,
    },
)
async def run_scenario(payload: ScenarioRequest):
    """
    🎚️ **Portfolio Risk Scenario Engine**

    Re-prices an entire book across a grid of shocked market states — the morning
    risk-screen workflow ("what does my P&L and Greeks look like if spot moves ±20%
    and vol shifts ±10 points?"). Uses **full revaluation** at every node (not a
    delta-gamma Taylor approximation), so the stress is exact.

    **Request**: a `portfolio` of signed positions on a *single underlying*, plus 1–2
    `shocks` axes. Each axis sweeps a factor (`spot`, `sigma`, `rate`, `carry`, `time`)
    either relatively (×(1+shock)) or absolutely (+shock).

    **Returns**: a `grid` of nodes — each carrying portfolio value, P&L vs. the unshocked
    base, and net Greeks (Σ quantity × per-contract Greek) — plus a `summary` flagging the
    worst-case node and max loss.

    **Cost**: billed as `positions × (nodes + 1)` pricing units; oversized requests are
    rejected up front with `413` rather than partially run.
    """
    positions = payload.portfolio
    n_positions = len(positions)

    # 1. Materialize the grid (Cartesian product of the axes, in request order).
    axis_values = [axis.shock_values() for axis in payload.shocks]
    node_combos = list(itertools.product(*axis_values))
    n_nodes = len(node_combos)

    # 2. Pre-flight cost check (base block + every node). Reject before doing work.
    total_valuations = n_positions * (n_nodes + 1)
    if total_valuations > MAX_SCENARIO_VALUATIONS:
        raise HTTPException(
            status_code=http_status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Scenario too large: {n_positions} positions × {n_nodes} nodes (+ base) = "
                f"{total_valuations} valuations exceeds the {MAX_SCENARIO_VALUATIONS} cap. "
                "Reduce grid density or split the book."
            ),
        )

    # 3. Netting Greeks across positions is only valid within one underlying, so a spot
    #    shock requires a single shared base spot.
    if any(axis.factor == "spot" for axis in payload.shocks):
        if len({p.config.s for p in positions}) > 1:
            raise HTTPException(
                status_code=422,
                detail="A spot shock requires all positions to share one underlying spot (single-underlying book).",
            )

    # 4. Build one contiguous config array: [base block] followed by one block per node.
    configs = (OptionConfig * total_valuations)()
    idx = 0
    for position in positions:
        _fill_config(configs[idx], position.config, {})
        idx += 1
    for combo in node_combos:
        for position in positions:
            overrides = {}
            for axis, shock in zip(payload.shocks, combo):
                field = FACTOR_FIELD[axis.factor]
                overrides[field] = _shocked_value(field, getattr(position.config, field), axis.mode, shock)
            _fill_config(configs[idx], position.config, overrides)
            idx += 1

    # 5. One batch pass through the engine (vega requested so it populates once implemented).
    status, greeks = fdm_price_batch(configs, total_valuations, True)
    if status == -99:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="The underlying proprietary quantitative processing core license has expired.",
        )
    if status != 0:
        raise HTTPException(status_code=500, detail="C++ core loop process failed during scenario evaluation.")

    # 6. Aggregate. Greeks are EXTENSIVE — net by weighted SUM (Σ quantity × greek), not average.
    quantities = [p.quantity for p in positions]
    base_value = sum(quantities[i] * greeks[i].price for i in range(n_positions))

    grid = []
    block = n_positions  # node blocks start right after the base block
    for combo in node_combos:
        value = sum(quantities[i] * greeks[block + i].price for i in range(n_positions))
        grid.append(
            ScenarioNode(
                shocks={axis.factor: shock for axis, shock in zip(payload.shocks, combo)},
                portfolio_value=value,
                pnl=value - base_value,
                delta=sum(quantities[i] * greeks[block + i].delta for i in range(n_positions)),
                gamma=sum(quantities[i] * greeks[block + i].gamma for i in range(n_positions)),
                vega=sum(quantities[i] * greeks[block + i].vega for i in range(n_positions)),
                theta=sum(quantities[i] * greeks[block + i].theta for i in range(n_positions)),
            )
        )
        block += n_positions

    worst_case = min(grid, key=lambda node: node.pnl)
    summary = ScenarioSummary(
        worst_case=worst_case,
        max_loss=worst_case.pnl,
        best_case_pnl=max(node.pnl for node in grid),
    )

    return ScenarioResponse(
        base_value=base_value,
        positions=n_positions,
        nodes=n_nodes,
        pricing_units=total_valuations,
        axes={axis.factor: values for axis, values in zip(payload.shocks, axis_values)},
        grid=grid,
        summary=summary,
    )
