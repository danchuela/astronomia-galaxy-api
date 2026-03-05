.PHONY: run test lint format typecheck precommit install

run:
	uv run uvicorn apps.api.main:app --reload --reload-dir apps --reload-dir packages

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run black .

typecheck:
	uv run mypy .

precommit:
	uv run pre-commit run --all-files

install:
	uv sync --group dev
