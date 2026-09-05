from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import MLModel, Prediction, User
from app.routers.auth import current_user

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/overview")
def overview(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return {
        "total_predictions": db.scalar(select(func.count(Prediction.id)).where(Prediction.user_id == user.id)) or 0,
        "active_model": db.scalar(select(func.count(MLModel.id)).where(MLModel.is_active.is_(True))) or 0,
    }


@router.get("/class-distribution")
def class_distribution(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(Prediction.predicted_class, func.count(Prediction.id))
        .where(Prediction.user_id == user.id)
        .group_by(Prediction.predicted_class)
    ).all()
    return [{"class": label, "count": count} for label, count in rows]


@router.get("/model-performance")
def model_performance(_user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(MLModel.id, MLModel.name, MLModel.version, MLModel.metrics_json).order_by(MLModel.created_at.desc())
    ).all()
    return [
        {
            "id": model_id,
            "name": name,
            "version": version,
            "metrics_json": {key: value for key, value in (metrics or {}).items() if not key.startswith("_")},
        }
        for model_id, name, version, metrics in rows
    ]
