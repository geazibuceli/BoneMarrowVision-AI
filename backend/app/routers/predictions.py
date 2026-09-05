from __future__ import annotations

import logging
import secrets
import warnings
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.audit import add_audit_event
from app.core.config import settings
from app.database import get_db
from app.inference import predict
from app.models import MLModel, Prediction, User
from app.routers.auth import current_user
from app.schemas import PredictionResponse

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_TYPES = {"image/png", "image/jpeg"}
FORMAT_EXTENSIONS = {"PNG": {".png"}, "JPEG": {".jpg", ".jpeg"}}
LOGGER = logging.getLogger(__name__)


def _prediction_response(prediction: Prediction) -> dict:
    return {
        "id": prediction.id,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "top_predictions_json": prediction.top_predictions_json,
        "gradcam_image_path": (
            f"/api/v1/predictions/{prediction.id}/gradcam" if prediction.gradcam_image_path else None
        ),
        "low_confidence": prediction.low_confidence,
        "created_at": prediction.created_at,
    }


def _stored_gradcam_path(prediction: Prediction) -> Path:
    if not prediction.gradcam_image_path:
        raise HTTPException(status_code=404, detail="Grad-CAM image not found")

    upload_root = settings.resolve_path(settings.upload_dir)
    stored_path = prediction.gradcam_image_path.replace("\\", "/")
    if stored_path.startswith("/media/gradcam/"):
        candidate = upload_root / "gradcam" / Path(stored_path).name
    else:
        candidate = settings.resolve_path(prediction.gradcam_image_path)
    try:
        candidate = candidate.resolve()
        candidate.relative_to(upload_root.resolve())
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Grad-CAM image not found") from error
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Grad-CAM image not found")
    return candidate


@router.post("", response_model=PredictionResponse, status_code=201)
async def create_prediction(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    try:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in ALLOWED_EXTENSIONS or file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail="Only PNG and JPEG images are accepted")
        content = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the configured size limit")
    if not content:
        raise HTTPException(status_code=415, detail="The uploaded image is empty")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                image_format = image.format
                width, height = image.size
                if width * height > settings.max_image_pixels:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Uploaded image exceeds the configured pixel limit",
                    )
                image.verify()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as error:
        raise HTTPException(status_code=415, detail="The uploaded file is not a valid PNG or JPEG image") from error
    if image_format not in FORMAT_EXTENSIONS or extension not in FORMAT_EXTENSIONS[image_format]:
        raise HTTPException(status_code=415, detail="Image contents do not match the file extension")

    active_model = db.scalar(select(MLModel).where(MLModel.is_active.is_(True)))
    if not active_model:
        raise HTTPException(status_code=503, detail="No active model is available for inference")

    upload_root = settings.resolve_path(settings.upload_dir)
    upload_root.mkdir(parents=True, exist_ok=True)
    inference_temp_root = settings.resolve_path(settings.inference_temp_dir)
    inference_temp_root.mkdir(parents=True, exist_ok=True)
    destination = inference_temp_root / f"bone-marrow-input-{secrets.token_hex(16)}{extension}"
    gradcam_path = upload_root / "gradcam" / f"{destination.stem}.png"
    await run_in_threadpool(destination.write_bytes, content)
    metrics = active_model.metrics_json or {}
    try:
        threshold = float(metrics.get("low_confidence_threshold", 0.60))
    except (TypeError, ValueError):
        threshold = 0.60
    if not 0 <= threshold <= 1:
        threshold = 0.60
    registered_digest = metrics.get("_checkpoint_sha256")
    if not isinstance(registered_digest, str) or len(registered_digest) != 64:
        registered_digest = None
    try:
        result = await run_in_threadpool(
            predict,
            destination,
            active_model.checkpoint_path,
            gradcam_path,
            threshold=threshold,
            expected_architecture=active_model.architecture,
            expected_sha256=registered_digest,
        )
    except Exception as error:
        LOGGER.exception("Inference failed for active model id=%s", active_model.id)
        destination.unlink(missing_ok=True)
        gradcam_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="The active model could not run inference") from error

    try:
        destination.unlink(missing_ok=True)
    except OSError as error:
        LOGGER.exception("Could not securely remove temporary inference input")
        gradcam_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Temporary inference input could not be removed") from error

    prediction = Prediction(
        user_id=user.id,
        model_id=active_model.id,
        gradcam_image_path=settings.project_relative_path(gradcam_path),
        **result,
    )
    db.add(prediction)
    try:
        db.flush()
        add_audit_event(
            db,
            user_id=user.id,
            action="prediction.created",
            resource_type="prediction",
            resource_id=prediction.id,
            metadata={"model_id": active_model.id},
        )
        db.commit()
    except SQLAlchemyError as error:
        LOGGER.exception("Could not persist prediction for user id=%s", user.id)
        db.rollback()
        destination.unlink(missing_ok=True)
        gradcam_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Prediction could not be saved") from error
    db.refresh(prediction)
    return _prediction_response(prediction)


@router.get("", response_model=list[PredictionResponse])
def list_predictions(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=1_000_000),
):
    predictions = db.scalars(
        select(Prediction)
        .where(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return [_prediction_response(prediction) for prediction in predictions]


@router.get("/{prediction_id}/gradcam", response_class=FileResponse)
def get_prediction_gradcam(
    prediction_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    prediction = db.scalar(select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user.id))
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return FileResponse(
        _stored_gradcam_path(prediction),
        media_type="image/png",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    prediction = db.scalar(select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user.id))
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return _prediction_response(prediction)
