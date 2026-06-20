from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from jose import jwt, JWTError
from app.config import settings


def get_user_id(request: Request) -> str:
    """Rate-limit / quota bucket key.

    Keys on the stable Cognito `sub` claim (an immutable UUID) so a user can't reset
    their bucket by changing email/username. Unauthenticated requests (e.g. /health)
    fall back to client IP.
    """
    # 1. Preferred source: the principal resolved + verified by the auth dependency.
    #    Available when the limit is checked after dependency resolution (decorator style).
    user = getattr(request.state, "user", None)
    if user is not None:
        return f"sub:{user.sub}"

    # 2. Fallback: derive `sub` directly from the bearer token. Needed when the limit is
    #    evaluated in the middleware stack, before dependencies run. The signature is NOT
    #    trusted here — get_current_user still verifies and rejects invalid tokens; this
    #    only needs a stable bucket key, and a forged `sub` gains nothing (the attacker
    #    just shares/throttles their own bucket and still fails auth at the endpoint).
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            sub = jwt.get_unverified_claims(auth_header[7:]).get("sub")
            if sub:
                return f"sub:{sub}"
        except JWTError:
            pass

    # 3. Anonymous request — bucket by source IP.
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_user_id)
rate_limit_string = f"{settings.RATE_LIMIT_RPM}/minute"
