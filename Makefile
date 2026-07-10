.PHONY: install test lint demo

# Create/refresh the local dev environment.
install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

# Run the automated test suite. pyproject sets pythonpath = ["src"],
# so no PYTHONPATH juggling is needed.
test:
	.venv/bin/pytest

# Lint the code with ruff.
lint:
	.venv/bin/ruff check .

# Run the market-pricing demo (no data download required).
demo:
	.venv/bin/python examples/market_demo.py
