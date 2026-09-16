FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/runtime.lock ./requirements.lock
COPY backend/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements.lock
COPY ml /app/ml
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /app/ml \
    && pip check

COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini
COPY backend/entrypoint.sh ./entrypoint.sh

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/uploads \
    && chown appuser:appuser /app /app/data /app/data/uploads

USER appuser
EXPOSE 8000
ENTRYPOINT ["sh", "/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
