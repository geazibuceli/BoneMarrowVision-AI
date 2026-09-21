from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index(
            "uq_users_single_admin",
            "role",
            unique=True,
            sqlite_where=text("role = 'admin'"),
            postgresql_where=text("role = 'admin'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(32), default="researcher")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class MLModel(Base):
    __tablename__ = "ml_models"
    __table_args__ = (
        Index(
            "uq_ml_models_single_active",
            "is_active",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(80))
    architecture: Mapped[str] = mapped_column(String(80))
    checkpoint_path: Mapped[str] = mapped_column(Text)
    config_path: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    model_id: Mapped[int | None] = mapped_column(ForeignKey("ml_models.id"), nullable=True)
    configuration_json: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset_version: Mapped[str] = mapped_column(String(160))
    train_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    test_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    model_id: Mapped[int | None] = mapped_column(ForeignKey("ml_models.id"), nullable=True)
    model: Mapped[MLModel | None] = relationship(lazy="joined")
    predicted_class: Mapped[str] = mapped_column(String(160))
    confidence: Mapped[float] = mapped_column(Float)
    top_predictions_json: Mapped[list] = mapped_column(JSON, default=list)
    gradcam_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    resource_type: Mapped[str] = mapped_column(String(100))
    resource_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
