.PHONY: venv sync test lock lint healthcheck bootstrap start-es index setup

venv:
	uv venv

sync:
	uv sync

test:
	uv run pytest

lock:
	uv lock

lint:
	uv run python -m ruff check .

healthcheck:
	uv run trialmatchai-healthcheck --config Matcher/config/config.json --start-es

bootstrap:
	bash scripts/bootstrap_data.sh

start-es:
	bash scripts/start_es.sh

index:
	bash scripts/index_data.sh

setup:
	bash setup.sh
