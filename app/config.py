from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OAUTH2_JWKS_URL: str = ""
    OAUTH2_AUDIENCE: str = ""
    OAUTH2_ISSUER: str = ""
    OAUTH2_AUTH_URL: str = ""
    OAUTH2_TOKEN_URL: str = ""

    RATE_LIMIT_RPM: int = 60

    CORS_ORIGINS: str = "*"
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    AWS_REGION: str = "ca-central-1"
    
    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
