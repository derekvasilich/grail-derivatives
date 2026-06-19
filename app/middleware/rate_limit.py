from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from app.config import settings

def get_user_id(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if user:
        return user.sub
    return get_remote_address(request)


limiter = Limiter(key_func=get_user_id)
rate_limit_string = f"{settings.RATE_LIMIT_RPM}/minute"
