from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.experiment_artifacts import (
    ARTIFACTS,
    artifact_path,
    read_artifact,
    run_directory,
    run_summary,
    runs_root,
)
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


@router.get("/runs")
def list_runs(
    _user: User = Depends(current_user),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0, le=1_000_000),
):
    root = runs_root()
    if not root.is_dir():
        return []
    summaries = []
    for path in sorted(root.iterdir(), key=lambda entry: entry.name, reverse=True):
        if not path.is_dir():
            continue
        try:
            directory = run_directory(path.name)
            if artifact_path(directory, "status.json").is_file():
                summaries.append(run_summary(directory))
        except HTTPException:
            # A malformed or inaccessible run must not hide all other runs.
            continue
    return summaries[offset:offset + limit]


@router.get("/runs/{run_id}")
def get_run(run_id: str, _user: User = Depends(current_user)):
    directory = run_directory(run_id)
    if not artifact_path(directory, "status.json").is_file():
        raise HTTPException(404, "Experiment run not found")
    return {"id": run_id, **{key: read_artifact(directory, key) for key in ARTIFACTS}}


@router.get("/runs/{run_id}/curves")
def get_run_curves(run_id: str, _user: User = Depends(current_user)):
    path = artifact_path(run_directory(run_id), "classifier_curves.png")
    if not path.is_file():
        raise HTTPException(404, "Learning curves are not available yet")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


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
