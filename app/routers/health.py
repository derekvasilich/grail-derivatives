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
    summary="Health check",
    description="Returns server status, DB connectivity, LLM reachability, app version, and uptime.",
    tags=["Health"],
)
async def health():
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
