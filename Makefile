SHELL := bash

.PHONY: help install lock test lint format readme build clean

help:
	@echo "Common commands:"
	@echo "  make install  - Install the package and dev dependencies (poetry install)"
	@echo "  make lock     - Update poetry.lock after changing pyproject.toml"
	@echo "  make test     - Run the test suite"
	@echo "  make lint     - Check code style with ruff"
	@echo "  make format   - Format code with black"
	@echo "  make readme   - Regenerate README.md from demo.ipynb"
	@echo "  make build    - Build the wheel and sdist into dist/"
	@echo "  make clean    - Remove build artifacts and caches"

install:
	poetry install

lock:
	poetry lock

test:
	poetry run pytest

lint:
	poetry run ruff check src tests

format:
	poetry run black src tests

readme:
	bash make_readme.sh

build: clean
	poetry build

clean:
	rm -rf dist/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
