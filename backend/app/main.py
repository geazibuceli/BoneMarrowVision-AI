from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import analytics, auth, experiments, health, models, predictions

app = FastAPI(
    title=settings.app_name,
    description="Research and educational bone marrow image classification API.",
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
