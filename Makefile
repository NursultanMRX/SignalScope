# Thin wrapper around tasks.py (works the same on Windows without make: `uv run python tasks.py <stage>`).
RUN = uv run python tasks.py

.PHONY: setup audit eda features tune train predict notebook site test test-all package all

setup:
	uv sync

audit features eda tune train predict notebook site test test-all package all:
	$(RUN) $@
