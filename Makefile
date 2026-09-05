PYTHON ?= python

.PHONY: install inspect prepare test lint build compose-config check up

install:
	$(PYTHON) -m pip install -e "ml[dev]"
	$(PYTHON) -m pip install -r backend/requirements-dev.txt

inspect:
	python ml/scripts/inspect_dataset.py --data-dir data/raw

prepare:
	python ml/scripts/prepare_data.py --config ml/configs/efficientnet_b0.yaml

test:
	cd ml && $(PYTHON) -m pytest tests
	cd backend && $(PYTHON) -m pytest tests
	cd frontend && npm test

lint:
	cd ml && ruff check src scripts tests
	cd backend && ruff check app tests alembic
	cd frontend && npm run lint

build:
	cd frontend && npm run build

compose-config:
	docker compose config --quiet

check: lint test build compose-config

up:
	docker compose up --build
