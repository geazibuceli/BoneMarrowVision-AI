from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Experiment, User
from app.routers.auth import current_user
from app.schemas import ExperimentResponse

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    _user: User = Depends(current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=1_000_000),
):
    statement = select(Experiment).order_by(Experiment.created_at.desc()).offset(offset).limit(limit)
    return list(db.scalars(statement))


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    experiment_id: int,
    _user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment
