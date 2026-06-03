# Task Completion Checklist
When a task is completed, you MUST ensure the following:
1. **Formatting and Linting**: The code is properly formatted and passes the linter. Use `just check`; it runs the repo's formatting and lint-fix step before validation.
2. **Type Checking**: All type hints are correct. `just check` runs strict mypy on `src/earnings_event_vol` and `tests`.
3. **Tests**: All tests must pass, and the overall test coverage must remain above 95%. `just check` runs pytest with coverage.
4. **All-in-one Check**: Ultimately, you should run `just check` to ensure all CI-like gates (formatting, linting, types, tests, docs, CLI checks) pass successfully.
5. **Credentials**: Ensure no direct API keys or secrets were accidentally hardcoded or stored in the repository. They must remain outside the repo.
