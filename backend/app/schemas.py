from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    full_name: str = Field(min_length=2, max_length=200)
    admin_bootstrap_token: str | None = Field(default=None, min_length=1, max_length=512, repr=False)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, email: EmailStr) -> str:
        return str(email).strip().lower()

    @field_validator("full_name")
    @classmethod
    def normalize_name(cls, full_name: str) -> str:
        normalized_name = " ".join(full_name.split())
        if len(normalized_name) < 2:
            raise ValueError("Full name must contain at least two non-whitespace characters")
        return normalized_name


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    full_name: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictionModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    version: str
    architecture: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    model: PredictionModelResponse | None = None
    predicted_class: str
    confidence: float
    top_predictions_json: list
    gradcam_image_path: str | None
    low_confidence: bool
    created_at: datetime
    disclaimer: str = (
        "Research and Educational Use Only. This system is not a medical device and must not be "
        "used for clinical diagnosis, treatment, or decision-making."
    )

    @field_validator("created_at")
    @classmethod
    def make_created_at_aware(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=UTC)
        return created_at


class ModelResponse(BaseModel):
    id: int
    name: str
    version: str
    architecture: Literal["baseline_cnn", "resnet50", "efficientnet_b0", "convnext_tiny"]
    metrics_json: dict[str, Any]
    checkpoint_sha256: str | None = None
    is_active: bool
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def make_created_at_aware(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=UTC)
        return created_at


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    model_id: int | None
    dataset_version: str
    configuration_json: dict[str, Any]
    train_metrics_json: dict[str, Any]
    validation_metrics_json: dict[str, Any]
    test_metrics_json: dict[str, Any]
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def make_created_at_aware(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=UTC)
        return created_at
