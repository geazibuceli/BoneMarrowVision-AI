from contextlib import asynccontextmanager
from functools import partial

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.retention import cleanup_expired_artifacts, run_periodic_cleanup
from app.routers import analytics, auth, experiments, health, models, predictions


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cleanup = partial(
        cleanup_expired_artifacts,
        upload_dir=settings.resolve_path(settings.upload_dir),
        inference_temp_dir=settings.resolve_path(settings.inference_temp_dir),
        gradcam_retention_days=settings.gradcam_retention_days,
        temp_retention_hours=settings.inference_temp_retention_hours,
    )
    await anyio.to_thread.run_sync(cleanup)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run_periodic_cleanup, cleanup, settings.retention_cleanup_interval_seconds)
        try:
            yield
        finally:
            tasks.cancel_scope.cancel()


app = FastAPI(
    title=settings.app_name,
    description="Research and educational bone marrow image classification API.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials="*" not in settings.cors_origin_list,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(models.router)
app.include_router(experiments.router)
app.include_router(analytics.router)


@app.get("/api/v1/disclaimer", tags=["health"])
def disclaimer():
    return {
        "text": (
            "Research and Educational Use Only. This system is not a medical device and must not "
            "be used for clinical diagnosis, treatment, or decision-making."
        )
    }
