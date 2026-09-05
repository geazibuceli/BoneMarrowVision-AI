import pickle
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.audit import add_audit_event
from app.core.config import settings
from app.database import get_db
from app.inference import checkpoint_sha256, load_model
from app.models import MLModel, User
from app.routers.auth import current_user, require_admin
from app.schemas import ModelResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=80)
    architecture: Literal["baseline_cnn", "resnet50", "efficientnet_b0"]
    checkpoint_path: str = Field(min_length=1, max_length=1000)
    config_path: str = Field(min_length=1, max_length=1000)
    metrics_json: dict[str, Any] = Field(default_factory=dict)


def _safe_registry_path(
    value: str,
    *,
    root: Path,
    allowed_suffixes: set[str],
    label: str,
) -> Path:
    candidate = settings.resolve_path(value)
    resolved_root = settings.resolve_path(root)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"{label} must be inside {resolved_root}") from error
    if candidate.suffix.lower() not in allowed_suffixes or not candidate.is_file():
        raise HTTPException(status_code=422, detail=f"{label} is missing or has an unsupported file type")
    return candidate


def _model_response(model: MLModel) -> dict[str, Any]:
    metrics = model.metrics_json or {}
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "architecture": model.architecture,
        "metrics_json": {key: value for key, value in metrics.items() if not key.startswith("_")},
        "checkpoint_sha256": metrics.get("_checkpoint_sha256"),
        "is_active": model.is_active,
        "created_at": model.created_at,
    }


@router.get("", response_model=list[ModelResponse])
def list_models(
    _user: User = Depends(current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=1_000_000),
):
    models = db.scalars(select(MLModel).order_by(MLModel.created_at.desc()).offset(offset).limit(limit))
    return [_model_response(model) for model in models]


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, _user: User = Depends(current_user), db: Session = Depends(get_db)):
    model = db.get(MLModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return _model_response(model)


@router.post("/register", response_model=ModelResponse, status_code=201)
def register_model(
    payload: ModelCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if any(key.startswith("_") for key in payload.metrics_json):
        raise HTTPException(status_code=422, detail="Metric names beginning with '_' are reserved")
    checkpoint_path = _safe_registry_path(
        payload.checkpoint_path,
        root=settings.model_dir,
        allowed_suffixes={".pt", ".pth"},
        label="Checkpoint",
    )
    config_path = _safe_registry_path(
        payload.config_path,
        root=settings.config_dir,
        allowed_suffixes={".json", ".yaml", ".yml"},
        label="Configuration",
    )
    try:
        digest = checkpoint_sha256(checkpoint_path)
        load_model(checkpoint_path, payload.architecture, digest)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as error:
        raise HTTPException(status_code=422, detail="Checkpoint is invalid or incompatible") from error
    model = MLModel(
        name=payload.name,
        version=payload.version,
        architecture=payload.architecture,
        checkpoint_path=settings.project_relative_path(checkpoint_path),
        config_path=settings.project_relative_path(config_path),
        metrics_json={**payload.metrics_json, "_checkpoint_sha256": digest},
    )
    db.add(model)
    try:
        db.flush()
        add_audit_event(
            db,
            user_id=admin.id,
            action="model.registered",
            resource_type="ml_model",
            resource_id=model.id,
            metadata={"architecture": model.architecture, "checkpoint_sha256": digest},
        )
        db.commit()
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(status_code=500, detail="Model could not be registered") from error
    db.refresh(model)
    return _model_response(model)


@router.post("/{model_id}/activate", response_model=ModelResponse)
def activate_model(
    model_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    model = db.get(MLModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    db.execute(update(MLModel).values(is_active=False))
    model.is_active = True
    add_audit_event(
        db,
        user_id=admin.id,
        action="model.activated",
        resource_type="ml_model",
        resource_id=model.id,
    )
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail="Another model activation completed concurrently; retry") from error
    db.refresh(model)
    return _model_response(model)
