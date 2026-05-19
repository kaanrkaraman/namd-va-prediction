default:
    @just --list

setup:
    uv sync
    chflags -R nohidden .venv 2>/dev/null || true

lint:
    uv run ruff check .
    uv run ruff format --check .

format:
    uv run ruff format .
    uv run ruff check --fix .

typecheck:
    uv run mypy

check: lint typecheck

download:
    uv run python scripts/01_download_data.py

cohort:
    uv run python scripts/02_build_cohort.py

train:
    uv run python scripts/03_train_xgboost.py

automl:
    uv run python scripts/04_train_autogluon.py

compare:
    uv run python scripts/05_compare_models.py

interpret:
    uv run python scripts/06_interpret_models.py

all: download cohort train automl compare interpret
