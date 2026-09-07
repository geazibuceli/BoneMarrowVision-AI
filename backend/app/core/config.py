from __future__ import annotations

import secrets
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _discover_project_root() -> Path:
    """Locate the repository both in a checkout and in the /app container layout."""
    if (BACKEND_ROOT / "ml").is_dir():
        return BACKEND_ROOT
    if (BACKEND_ROOT.parent / "ml").is_dir():
        return BACKEND_ROOT.parent
    return BACKEND_ROOT


PROJECT_ROOT = _discover_project_root()


def _default_database_url() -> str:
    database_path = (PROJECT_ROOT / "bone_marrow_vision.db").resolve()
    return f"sqlite:///{database_path.as_posix()}"


class Settings(BaseSettings):
    app_name: str = "BoneMarrowVision AI"
    app_environment: Literal["development", "test", "production"] = "development"
    project_root: Path = PROJECT_ROOT
    database_url: str = Field(default_factory=_default_database_url)
    jwt_secret: str = Field(default="", repr=False)
    jwt_issuer: str = "bone-marrow-vision-api"
    jwt_audience: str = "bone-marrow-vision-web"
    jwt_expire_minutes: int = Field(default=60, ge=5, le=24 * 60)
    admin_bootstrap_token: SecretStr | None = Field(default=None, repr=False)
    upload_dir: Path = Path("data/uploads")
    inference_temp_dir: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()))
    model_dir: Path = Path("models")
    config_dir: Path = Path("ml/configs")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1, le=100_000_000)
    max_checkpoint_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1)
    auth_rate_limit: int = Field(default=30, ge=1, le=10_000)
    auth_rate_window_seconds: int = Field(default=60, ge=1, le=3_600)
    prediction_rate_limit: int = Field(default=30, ge=1, le=10_000)
    prediction_rate_window_seconds: int = Field(default=60, ge=1, le=3_600)
    gradcam_retention_days: int = Field(default=30, ge=1, le=3_650)
    inference_temp_retention_hours: int = Field(default=24, ge=1, le=8_760)
    trust_proxy_headers: bool = False
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_security_settings(self) -> Settings:
        self.project_root = self.project_root.expanduser().resolve()

        if self.database_url.startswith("sqlite:///./"):
            relative_path = self.database_url.removeprefix("sqlite:///./")
            absolute_path = (self.project_root / relative_path).resolve()
            self.database_url = f"sqlite:///{absolute_path.as_posix()}"

        weak_secrets = {
            "change-this-in-production",
            "local-development-secret",
            "replace-with-a-long-random-secret",
        }
        if not self.jwt_secret:
            if self.app_environment == "production":
                raise ValueError("JWT_SECRET must be explicitly configured in production")
            self.jwt_secret = secrets.token_urlsafe(48)
        elif self.app_environment == "production" and (
            len(self.jwt_secret.strip()) < 32 or self.jwt_secret in weak_secrets
        ):
            raise ValueError("JWT_SECRET must be at least 32 characters and not use an example value")

        if (
            self.app_environment == "production"
            and self.admin_bootstrap_token is not None
            and len(self.admin_bootstrap_token.get_secret_value().strip()) < 32
        ):
            raise ValueError("ADMIN_BOOTSTRAP_TOKEN must be at least 32 characters in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def project_relative_path(self, path: str | Path) -> str:
        resolved = self.resolve_path(path)
        try:
            return resolved.relative_to(self.project_root).as_posix()
        except ValueError:
            return str(resolved)


settings = Settings()
