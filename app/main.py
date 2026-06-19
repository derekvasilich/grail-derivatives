import structlog
import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.middleware.rate_limit import limiter
from app.config import settings
from app.auth.jwt import get_current_user
from app.routers import health, pricing

API_DESCRIPTION = """
Production-grade options valuation pipeline powered by **C++ OpenMP core loops**.

The engine prices Vanilla (European) contracts and path-dependent Exotics (American,
Bermudan, single & double Barrier) by solving the Black–Scholes PDE on a 2D asset-time
mesh via **Crank–Nicolson** discretization with **Rannacher damping** and an `O(N)`
Thomas-algorithm tridiagonal solver.

### Authentication
All `/v1/pricing/*` routes require a valid **AWS Cognito OAuth2 JWT** sent as
`Authorization: Bearer <token>`. The `/v1/health` probe is public.
In local dev (empty `OAUTH2_JWKS_URL`) any unsigned JWT with a `sub` claim is accepted.

### Response formats
Routes return JSON Greeks, raw `float64` binary surfaces, or rendered PNG charts —
see each endpoint's responses for the exact `Content-Type` and headers.
"""

OPENAPI_TAGS = [
    {
        "name": "Pricing",
        "description": "Option valuation endpoints — single, batch, zero-copy binary, full "
                       "surface grid, and rendered 3D chart. All require a bearer token.",
    },
    {
        "name": "Health",
        "description": "Public liveness & dependency probe. No authentication required.",
    },
]

app = FastAPI(
    title="Grail Derivatives: Black-Scholes HPC FDM Pricing Engine",
    description=API_DESCRIPTION,
    version="1.0.0-BETA",
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
        "clientId": settings.OAUTH2_AUDIENCE,
    },
)

# CORS
origins = settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

PREFIX = "/v1"

# /health is wired in without auth so it stays a public liveness probe.
app.include_router(health.router, prefix=PREFIX)
# Every pricing route (current and future) requires a valid JWT via this single
app.include_router(
    pricing.router,
    prefix=PREFIX,
    dependencies=[Depends(get_current_user)],
)

# ============================================================
# 3. WEB ENDPOINT ROUTING DRIVERS
# ============================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    if hasattr(exc, "status_code") and hasattr(exc, "detail"):
        detail = exc.detail
        if isinstance(detail, dict):
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(detail), "details": {}}},
        )
    structlog.get_logger().error("unhandled_exception", exc=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal server error", "details": {}}},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
