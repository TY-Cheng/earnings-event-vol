# Code Style and Conventions
- **Linter**: `ruff` is used for linting and formatting. Line length is 100, targeting Python 3.11. Selected rule sets: `E`, `F`, `I`, `UP`, `B`, `SIM`.
- **Type Checking**: `mypy` is used with strict mode enabled (`strict = true`).
- **Testing**: `pytest` is used with `pytest-cov`. Tests must maintain at least a 95% coverage.
- **Formatting Command**: `uv run ruff format src tests`
- **Linting Command**: `uv run ruff check src tests` (add `--fix` to auto-fix)
- **Typing Command**: `uv run mypy src tests`
