from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from alembic import command
from app.core.config import BACKEND_ROOT, settings
from app.database import Base


def _database_url(database_path: Path) -> str:
    return f"sqlite:///{database_path.as_posix()}"


def _alembic_config() -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    return config


def test_initial_migration_creates_fresh_database(workspace_tmp_path: Path, monkeypatch):
    database_url = _database_url(workspace_tmp_path / "fresh.db")
    monkeypatch.setattr(settings, "database_url", database_url)

    command.upgrade(_alembic_config(), "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert set(Base.metadata.tables).issubset(inspector.get_table_names())
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == "0002_account_recovery"
    engine.dispose()


def test_recovery_migration_preserves_existing_accounts(workspace_tmp_path, monkeypatch):
    database_url = _database_url(workspace_tmp_path / "upgrade.db")
    monkeypatch.setattr(settings, "database_url", database_url)
    config = _alembic_config()
    command.upgrade(config, "0001_initial")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO users (id,email,password_hash,full_name,role,created_at,updated_at) "
            "VALUES (1,'preserved@example.com','existing-hash','Existing User','researcher',"
            "CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
        ))
    command.upgrade(config, "head")
    with engine.connect() as connection:
        row = connection.execute(text("SELECT email,password_hash,session_version,reset_token_hash FROM users")).one()
        assert tuple(row) == ("preserved@example.com", "existing-hash", 0, None)
    command.downgrade(config, "0001_initial")
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT password_hash FROM users")) == "existing-hash"
    engine.dispose()


def test_initial_migration_adopts_compatible_legacy_database(workspace_tmp_path: Path, monkeypatch):
    database_url = _database_url(workspace_tmp_path / "legacy.db")
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX uq_users_single_admin")
        connection.exec_driver_sql("DROP INDEX uq_ml_models_single_active")
        connection.exec_driver_sql("ALTER TABLE experiments ADD COLUMN mlflow_run_id VARCHAR(160)")
        connection.exec_driver_sql("ALTER TABLE predictions ADD COLUMN input_image_path TEXT")
    assert "alembic_version" not in inspect(engine).get_table_names()
    engine.dispose()

    monkeypatch.setattr(settings, "database_url", database_url)
    command.upgrade(_alembic_config(), "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    index_names = {index["name"] for index in inspector.get_indexes("users")}
    assert {"ix_users_email", "uq_users_single_admin"}.issubset(index_names)
    model_index_names = {index["name"] for index in inspector.get_indexes("ml_models")}
    assert "uq_ml_models_single_active" in model_index_names
    assert "mlflow_run_id" not in {column["name"] for column in inspector.get_columns("experiments")}
    assert "input_image_path" not in {column["name"] for column in inspector.get_columns("predictions")}
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == "0002_account_recovery"
    engine.dispose()
