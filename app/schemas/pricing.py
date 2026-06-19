from pydantic import BaseModel, Field
from typing import List

# ============================================================
# 2. PYDANTIC REQUEST/RESPONSE VALIDATION SCHEMAS
# ============================================================
class OptionConfigSchema(BaseModel):
    deriv: int = Field(..., description="0=VanillaCall, 1=VanillaPut, 2=AmericanCall, 3=AmericanPut, 4=BermudanCall, 5=BermudanPut")
    Tn: int = Field(1000, description="Number of temporal time grid increments")
    top: int = 0
    bottom: int = 0
    right: int = 0
    left: int = 0
    time: float = Field(1.0, description="Time to maturity in years (T)")
    h: float = Field(1, description="Spatial mesh coordinate spacing thickness (dx)")
    r: float = Field(0.1, description="Risk-free interest rate constant multiplier")
    sigma: float = Field(0.5, description="Asset implied volatility percentage index")
    s: float = Field(100, description="Underlying asset spot market price (S)")
    k: float = Field(110, description="Option contractual strike target price (K)")
    q: float = Field(0.0, description="Continuous asset dividend payout yield percentage")

class CompactGreeksResponse(BaseModel):
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    Tn: int
    Xm: int

class GridPricingResponse(BaseModel):
    status: int
    greeks: CompactGreeksResponse
    price_matrix: List[List[float]]