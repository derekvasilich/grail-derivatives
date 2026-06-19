from pydantic import BaseModel, ConfigDict, Field
from typing import List

# ============================================================
# 2. PYDANTIC REQUEST/RESPONSE VALIDATION SCHEMAS
# ============================================================
class OptionConfigSchema(BaseModel):
    """A single option contract specification routed down to the C++ FDM core.

    The integer `deriv` flag selects the payoff/exercise style; the remaining fields
    define the market state and the finite-difference mesh resolution.
    """

    deriv: int = Field(
        ...,
        ge=0,
        le=5,
        description="Derivative type flag: 0=VanillaCall, 1=VanillaPut, 2=AmericanCall, "
                    "3=AmericanPut, 4=BermudanCall, 5=BermudanPut.",
        examples=[4],
    )
    Tn: int = Field(
        1000,
        gt=0,
        description="Number of temporal time-grid increments (matrix time-steps). "
                    "Higher values increase accuracy at the cost of compute.",
        examples=[1000],
    )
    top: int = Field(0, description="Reserved exotic-barrier boundary flag (upper). Leave 0 for vanilla/American/Bermudan.")
    bottom: int = Field(0, description="Reserved exotic-barrier boundary flag (lower). Leave 0 for vanilla/American/Bermudan.")
    right: int = Field(0, description="Reserved exotic-barrier boundary flag (right edge). Leave 0 unless pricing barriers.")
    left: int = Field(0, description="Reserved exotic-barrier boundary flag (left edge). Leave 0 unless pricing barriers.")
    time: float = Field(1.0, gt=0, description="Time to maturity in years (T).", examples=[1.0])
    h: float = Field(1, gt=0, description="Spatial mesh coordinate spacing thickness (dx) of the asset grid.", examples=[1.0])
    r: float = Field(0.1, description="Continuously-compounded risk-free interest rate (decimal, e.g. 0.1 = 10%).", examples=[0.1])
    sigma: float = Field(0.5, gt=0, description="Asset implied volatility (decimal, e.g. 0.5 = 50%).", examples=[0.5])
    s: float = Field(100, gt=0, description="Underlying asset spot market price (S).", examples=[100.0])
    k: float = Field(110, gt=0, description="Option contractual strike target price (K).", examples=[110.0])
    q: float = Field(0.0, description="Continuous dividend payout yield (decimal, e.g. 0.02 = 2%).", examples=[0.0])

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "deriv": 4,
                    "s": 100.0,
                    "k": 110.0,
                    "time": 1.0,
                    "sigma": 0.5,
                    "r": 0.1,
                    "q": 0.0,
                    "Tn": 1000,
                    "h": 1.0,
                }
            ]
        }
    )


class CompactGreeksResponse(BaseModel):
    """The compact risk-metrics block returned by the single and batch pricing routes."""

    price: float = Field(..., description="Fair option premium at the spot price S.", examples=[14.231])
    delta: float = Field(..., description="∂V/∂S — sensitivity of premium to a unit move in the underlying.", examples=[0.512])
    gamma: float = Field(..., description="∂²V/∂S² — rate of change of delta (convexity).", examples=[0.018])
    theta: float = Field(..., description="∂V/∂t — time decay of the premium per year.", examples=[-6.42])
    vega: float = Field(..., description="∂V/∂σ — sensitivity to volatility. 0.0 unless calculate_vega=true.", examples=[38.7])
    Tn: int = Field(..., description="Temporal grid rows the engine resolved for the solve.", examples=[1000])
    Xm: int = Field(..., description="Spatial (asset) grid columns the engine resolved for the solve.", examples=[201])


class GridPricingResponse(BaseModel):
    status: int
    greeks: CompactGreeksResponse
    price_matrix: List[List[float]]
