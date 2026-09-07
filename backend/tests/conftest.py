import shutil
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import BACKEND_ROOT
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
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Generator[TestClient]:
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
