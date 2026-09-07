FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
COPY backend/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -r requirements.txt
COPY ml /app/ml
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir \
        "scikit-learn>=1.4" "numpy>=1.26" "PyYAML>=6.0" "tqdm>=4.66" \
        "matplotlib>=3.8" "seaborn>=0.13" \
    && pip install --no-cache-dir --no-deps -e /app/ml

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
