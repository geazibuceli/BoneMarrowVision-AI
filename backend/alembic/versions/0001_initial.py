"""Create the initial research platform schema.

The first local releases created tables directly through SQLAlchemy and did not
record an Alembic revision. The guarded creation below adopts those compatible
databases while still rejecting a partial legacy table that would otherwise be
silently stamped as current.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

from alembic import context, op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None

TABLE_COLUMNS = {
    "users": {"id", "email", "password_hash", "full_name", "role", "created_at", "updated_at"},
    "ml_models": {
        "id",
        "name",
        "version",
        "architecture",
        "checkpoint_path",
        "config_path",
        "metrics_json",
        "is_active",
        "created_at",
    },
    "experiments": {
        "id",
        "name",
        "model_id",
        "configuration_json",
        "dataset_version",
        "train_metrics_json",
        "validation_metrics_json",
        "test_metrics_json",
        "created_at",
    },
    "predictions": {
        "id",
        "user_id",
        "model_id",
        "predicted_class",
        "confidence",
        "top_predictions_json",
        "gradcam_image_path",
        "low_confidence",
        "created_at",
    },
    "audit_logs": {
        "id",
        "user_id",
        "action",
        "resource_type",
        "resource_id",
        "metadata_json",
        "created_at",
    },
}


def _online_inspector() -> Inspector | None:
    if context.is_offline_mode():
        return None
    return sa.inspect(op.get_bind())


def _validate_legacy_table(inspector: Inspector | None, table_name: str) -> None:
    if inspector is None:
        return
    actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
    missing_columns = TABLE_COLUMNS[table_name] - actual_columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"Legacy table '{table_name}' is incompatible; missing columns: {missing}")


def _drop_legacy_column(inspector: Inspector | None, table_name: str, column_name: str) -> None:
    if inspector is None:
        return
    column_names = {column["name"] for column in inspector.get_columns(table_name)}
    if column_name in column_names:
        with op.batch_alter_table(table_name) as batch_operation:
            batch_operation.drop_column(column_name)


def upgrade() -> None:
    inspector = _online_inspector()
    existing_tables = set(inspector.get_table_names()) if inspector is not None else set()

    if "users" in existing_tables:
        _validate_legacy_table(inspector, "users")
    else:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("email", sa.String(320), nullable=False),
            sa.Column("password_hash", sa.Text(), nullable=False),
            sa.Column("full_name", sa.String(200), nullable=False),
            sa.Column("role", sa.String(32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

    user_indexes = (
        {index["name"] for index in inspector.get_indexes("users")}
        if inspector is not None and "users" in existing_tables
        else set()
    )
    if "ix_users_email" not in user_indexes:
        op.create_index("ix_users_email", "users", ["email"], unique=True)
    if "uq_users_single_admin" not in user_indexes:
        op.create_index(
            "uq_users_single_admin",
            "users",
            ["role"],
            unique=True,
            sqlite_where=sa.text("role = 'admin'"),
            postgresql_where=sa.text("role = 'admin'"),
        )

    if "ml_models" in existing_tables:
        _validate_legacy_table(inspector, "ml_models")
    else:
        op.create_table(
            "ml_models",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("name", sa.String(200), nullable=False),
            sa.Column("version", sa.String(80), nullable=False),
            sa.Column("architecture", sa.String(80), nullable=False),
            sa.Column("checkpoint_path", sa.Text(), nullable=False),
            sa.Column("config_path", sa.Text(), nullable=False),
            sa.Column("metrics_json", sa.JSON(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    model_indexes = (
        {index["name"] for index in inspector.get_indexes("ml_models")}
        if inspector is not None and "ml_models" in existing_tables
        else set()
    )
    if "uq_ml_models_single_active" not in model_indexes:
        op.create_index(
            "uq_ml_models_single_active",
            "ml_models",
            ["is_active"],
            unique=True,
            sqlite_where=sa.text("is_active = 1"),
            postgresql_where=sa.text("is_active"),
        )

    if "experiments" in existing_tables:
        _validate_legacy_table(inspector, "experiments")
        _drop_legacy_column(inspector, "experiments", "mlflow_run_id")
    else:
        op.create_table(
            "experiments",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("name", sa.String(200), nullable=False),
            sa.Column("model_id", sa.Integer(), sa.ForeignKey("ml_models.id"), nullable=True),
            sa.Column("configuration_json", sa.JSON(), nullable=False),
            sa.Column("dataset_version", sa.String(160), nullable=False),
            sa.Column("train_metrics_json", sa.JSON(), nullable=False),
            sa.Column("validation_metrics_json", sa.JSON(), nullable=False),
            sa.Column("test_metrics_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "predictions" in existing_tables:
        _validate_legacy_table(inspector, "predictions")
        _drop_legacy_column(inspector, "predictions", "input_image_path")
    else:
        op.create_table(
            "predictions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("model_id", sa.Integer(), sa.ForeignKey("ml_models.id"), nullable=True),
            sa.Column("predicted_class", sa.String(160), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("top_predictions_json", sa.JSON(), nullable=False),
            sa.Column("gradcam_image_path", sa.Text(), nullable=True),
            sa.Column("low_confidence", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "audit_logs" in existing_tables:
        _validate_legacy_table(inspector, "audit_logs")
    else:
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("action", sa.String(100), nullable=False),
            sa.Column("resource_type", sa.String(100), nullable=False),
            sa.Column("resource_id", sa.String(100), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade() -> None:
    inspector = _online_inspector()
    existing_tables = set(inspector.get_table_names()) if inspector is not None else set(TABLE_COLUMNS)
    for table_name in ("audit_logs", "predictions", "experiments", "ml_models"):
        if table_name in existing_tables:
            op.drop_table(table_name)
    if "users" in existing_tables:
        op.drop_table("users")
