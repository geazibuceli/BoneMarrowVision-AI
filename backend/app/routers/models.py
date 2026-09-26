import pickle
from datetime import UTC
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
from app.models import AuditLog, MLModel, User
from app.routers.auth import current_user, require_admin
from app.schemas import ModelResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=80)
    architecture: Literal["baseline_cnn", "resnet50", "efficientnet_b0", "convnext_tiny"]
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
    models = db.scalars(
        select(MLModel).order_by(MLModel.created_at.desc(), MLModel.id.desc()).offset(offset).limit(limit)
    )
    return [_model_response(model) for model in models]


@router.get("/activation-history")
def activation_history(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=1_000_000),
):
    events = db.scalars(
        select(AuditLog).where(AuditLog.action == "model.activated", AuditLog.resource_type == "ml_model")
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).offset(offset).limit(limit)
    ).all()
    model_ids = {int(event.resource_id) for event in events if (event.resource_id or "").isdigit()}
    models = {model.id: model for model in db.scalars(select(MLModel).where(MLModel.id.in_(model_ids)))}
    history = []
    for event in events:
        model_id = int(event.resource_id) if (event.resource_id or "").isdigit() else None
        model = models.get(model_id)
        metadata = event.metadata_json or {}
        history.append({
            "id": event.id,
            "model_id": model_id if model else None,
            "model_name": metadata.get("model_name", model.name if model else "Unavailable model"),
            "model_version": metadata.get("model_version", model.version if model else None),
            "created_at": event.created_at.replace(tzinfo=UTC) if event.created_at.tzinfo is None else event.created_at,
        })
    return history


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
    try:
        load_model(model.checkpoint_path, model.architecture, (model.metrics_json or {}).get("_checkpoint_sha256"))
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as error:
        raise HTTPException(
            status_code=422, detail="Checkpoint is unavailable or incompatible; active model unchanged",
        ) from error
    if model.is_active:
        return _model_response(model)
    try:
        db.execute(update(MLModel).values(is_active=False))
        model.is_active = True
        add_audit_event(
            db,
            user_id=admin.id,
            action="model.activated",
            resource_type="ml_model",
            resource_id=model.id,
            metadata={"model_name": model.name, "model_version": model.version},
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail="Another model activation completed concurrently; retry") from error
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(status_code=500, detail="Model could not be activated") from error
    db.refresh(model)
    return _model_response(model)
