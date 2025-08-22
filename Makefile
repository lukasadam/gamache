.PHONY: help bootstrap setup test lint format types precommit docs docs-serve clean

help:
	@echo "Common tasks:"
	@echo "  make bootstrap      # create venv, install dev/test, pre-commit, (docs if configured)"
	@echo "  make setup          # just dev/test deps"
	@echo "  make test           # run pytest"
	@echo "  make lint           # ruff/black/isort checks"
	@echo "  make format         # apply ruff/black/isort fixes"
	@echo "  make precommit      # run hooks on all files"
	@echo "  make docs           # build docs (if mkdocs present)"
	@echo "  make docs-serve     # serve docs locally"
	@echo "  make clean          # remove caches and build artifacts"

bootstrap:
	./scripts/bootstrap.sh --with-docs auto

setup:
	uv venv
	. .venv/bin/activate && uv pip install -e '.[dev,test]'

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run black --check .
	uv run isort --check-only .

format:
	uv run ruff --fix .
	uv run ruff format .
	uv run black .
	uv run isort .

precommit:
	uv pip install pre-commit
	uv run pre-commit install
	uv run pre-commit run --all-files

docs:
	uv pip install -e '.[docs]'
	uv run mkdocs build --strict

docs-serve:
	uv pip install -e '.[docs]'
	uv run mkdocs serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build site _site
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
