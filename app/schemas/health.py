from pydantic import BaseModel
from typing import Dict

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    authentication: str
