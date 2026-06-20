"""Request/response schemas for the portfolio risk-scenario engine.

A scenario run shocks one or two **market factors** across a grid and re-prices the
whole book at every node, returning portfolio P&L and net Greeks per node plus a
worst-case summary. See app/routers/risk.py for the execution pipeline.
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.pricing import OptionConfigSchema

FactorName = Literal["spot", "sigma", "rate", "carry", "time"]
ShockMode = Literal["relative", "absolute"]

# Maps a market factor onto the OptionConfig field it perturbs.
FACTOR_FIELD: Dict[str, str] = {
    "spot": "s",
    "sigma": "sigma",
    "rate": "r",
    "carry": "q",
    "time": "time",
}


class ShockAxis(BaseModel):
    """One axis of the scenario grid: a market factor swept over a set of shocks.

    Specify the shocks either as a range (`from`/`to`/`step`) or explicit `values`.
    `relative` mode multiplies the base value by `(1 + shock)`; `absolute` adds `shock`.
    """

    factor: FactorName = Field(..., description="Market factor to shock.")
    mode: ShockMode = Field("absolute", description="`relative` = ×(1+shock); `absolute` = +shock.")
    from_: Optional[float] = Field(None, alias="from", description="Range start (inclusive).")
    to: Optional[float] = Field(None, description="Range end (inclusive).")
    step: Optional[float] = Field(None, description="Range step (non-zero).")
    values: Optional[List[float]] = Field(None, description="Explicit shock values (alternative to range).")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _check_spec(self):
        if self.values is None:
            if self.from_ is None or self.to is None or self.step is None:
                raise ValueError("shock axis needs either 'values' or all of 'from'+'to'+'step'")
            if self.step == 0:
                raise ValueError("'step' cannot be 0")
            if (self.to - self.from_) * self.step < 0:
                raise ValueError("'step' sign must move 'from' toward 'to'")
        elif len(self.values) == 0:
            raise ValueError("'values' cannot be empty")
        return self

    def shock_values(self) -> List[float]:
        """Materialize the concrete list of shock magnitudes for this axis."""
        if self.values is not None:
            return list(self.values)
        n = int(round((self.to - self.from_) / self.step))
        return [round(self.from_ + i * self.step, 10) for i in range(n + 1)]


class PortfolioPosition(BaseModel):
    config: OptionConfigSchema = Field(..., description="The option contract definition.")
    quantity: float = Field(..., description="Signed position size; negative = short.", examples=[10])


class ScenarioRequest(BaseModel):
    portfolio: List[PortfolioPosition] = Field(..., min_length=1, description="Positions in one underlying.")
    shocks: List[ShockAxis] = Field(..., min_length=1, max_length=2, description="1 axis (ladder) or 2 (matrix).")

    @model_validator(mode="after")
    def _validate_axes(self):
        factors = [ax.factor for ax in self.shocks]
        if len(set(factors)) != len(factors):
            raise ValueError("each shock axis must use a distinct factor")
        for ax in self.shocks:
            if ax.factor == "spot" and ax.mode == "relative" and min(ax.shock_values()) <= -1:
                raise ValueError("a relative spot shock must be > -1 (a -100% move zeroes the spot)")
        return self


class ScenarioNode(BaseModel):
    shocks: Dict[str, float] = Field(..., description="The shock applied per factor at this node.")
    portfolio_value: float = Field(..., description="Σ quantity_i × price_i at this node.")
    pnl: float = Field(..., description="portfolio_value(node) − portfolio_value(base).")
    delta: float = Field(..., description="Net portfolio delta (Σ quantity_i × delta_i).")
    gamma: float = Field(..., description="Net portfolio gamma.")
    vega: float = Field(..., description="Net portfolio vega.")
    theta: float = Field(..., description="Net portfolio theta.")


class ScenarioSummary(BaseModel):
    worst_case: ScenarioNode = Field(..., description="The grid node with the lowest P&L.")
    max_loss: float = Field(..., description="Lowest P&L across all nodes (negative = loss).")
    best_case_pnl: float = Field(..., description="Highest P&L across all nodes.")


class ScenarioResponse(BaseModel):
    base_value: float = Field(..., description="Unshocked portfolio mark.")
    positions: int
    nodes: int
    pricing_units: int = Field(..., description="Valuations performed = positions × (nodes + 1 base block).")
    axes: Dict[str, List[float]] = Field(..., description="The materialized shock values per factor.")
    grid: List[ScenarioNode]
    summary: ScenarioSummary
