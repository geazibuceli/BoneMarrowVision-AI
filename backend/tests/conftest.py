import os
import shutil
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import BACKEND_ROOT
from app.core.rate_limit import rate_limiter
from app.database import Base, get_db
from app.main import app


@pytest.fixture
def workspace_tmp_path() -> Generator[Path]:
    """Use a normal workspace directory; the managed Windows sandbox blocks pytest's chmod-based temp root."""
    test_directory = BACKEND_ROOT / f"backend_test_{uuid4().hex}"
    test_directory.mkdir()
    try:
        yield test_directory
    finally:
        if test_directory.parent != BACKEND_ROOT:
            raise RuntimeError("Refusing to clean an unexpected test directory")
        shutil.rmtree(test_directory, ignore_errors=False)


@pytest.fixture
def session_factory() -> Generator[sessionmaker[Session]]:
    database_url = os.environ.get("TEST_DATABASE_URL", "sqlite://")
    is_sqlite = database_url.startswith("sqlite")
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if is_sqlite else {},
        poolclass=StaticPool if is_sqlite else None,
    )
    if is_sqlite:
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
    else:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE audit_logs, predictions, experiments, "
                    "ml_models, users RESTART IDENTITY CASCADE"
                )
            )
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        if not is_sqlite:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "TRUNCATE TABLE audit_logs, predictions, experiments, "
                        "ml_models, users RESTART IDENTITY CASCADE"
                    )
                )
        else:
            Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Generator[TestClient]:
    rate_limiter.clear()
    def override_get_db() -> Generator[Session]:
        database = session_factory()
        try:
            yield database
        finally:
            database.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
        rate_limiter.clear()
