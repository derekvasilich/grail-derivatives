import time
from fastapi import APIRouter
import httpx
from app.schemas.health import HealthResponse
from app.config import settings

router = APIRouter()
_start_time = time.time()

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness & dependency probe",
    response_description="Server status, app version, uptime, and upstream auth-provider reachability.",
    responses={
        200: {
            "description": "Service is live. `authentication` reflects whether the Cognito "
                           "JWKS endpoint was reachable — `\"ok\"`, or `\"error: <reason>\"` if not.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "version": "1.0.0-BETA",
                        "uptime_seconds": 12.34,
                        "authentication": "ok",
                    }
                }
            },
        }
    },
)
async def health():
    """
    ❤️ **Liveness & Dependency Probe**

    Public, unauthenticated readiness check intended for load-balancer health checks and
    uptime monitors. Reports the running `version`, process `uptime_seconds`, and actively
    pings the upstream **Cognito JWKS** endpoint so you can detect identity-provider
    outages before they surface as `401`s on the pricing routes.

    The `authentication` field is `"ok"` when JWKS is reachable, or `"error: <reason>"`
    otherwise — the probe still returns `200` so liveness and dependency state stay distinct.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(settings.OAUTH2_JWKS_URL, timeout=10)
            resp.raise_for_status()
        auth_status = "ok"
    except Exception as e:
        auth_status = f"error: {e}"

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        uptime_seconds=round(time.time() - _start_time, 2),
        authentication=auth_status,
    )
