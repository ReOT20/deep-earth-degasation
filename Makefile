.PHONY: check lint type test format

check: lint type test

lint:
	ruff check .
	ruff format --check .

type:
	basedpyright

test:
	pytest

format:
	ruff format .
	ruff check . --fix
