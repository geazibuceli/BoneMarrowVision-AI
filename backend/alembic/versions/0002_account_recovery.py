"""Add single-use password recovery, session revocation, and history index."""

import sqlalchemy as sa

from alembic import op

revision = "0002_account_recovery"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    # Compatible with legacy databases adopted by the initial migration.
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("users")}
    for column in (
        sa.Column("session_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reset_token_hash", sa.String(64), nullable=True),
        sa.Column("reset_token_expires_at", sa.DateTime(timezone=True), nullable=True),
    ):
        if column.name not in columns:
            op.add_column("users", column)
    indexes = {index["name"] for index in inspector.get_indexes("predictions")}
    if "ix_predictions_owner_created" not in indexes:
        op.create_index("ix_predictions_owner_created", "predictions", ["user_id", "created_at", "id"])


def downgrade():
    op.drop_index("ix_predictions_owner_created", table_name="predictions")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("reset_token_expires_at")
        batch.drop_column("reset_token_hash")
        batch.drop_column("session_version")
