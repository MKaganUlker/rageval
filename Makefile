install:
	pip install -e ".[dev]"

run:
	rageval run --config configs/baseline.yaml

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src

check: lint typecheck test
