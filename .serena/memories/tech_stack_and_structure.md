# Tech Stack

- **Language**: Python `>=3.11,<3.14`
- **Package manager**: `uv`
- **Environment policy**: `.env` or the shell sets `UV_PROJECT_ENVIRONMENT`
  outside the repo. Current WSL2/CUDA venv:
  `/home/tycheng/.venvs/earnings-event-vol`. Do not use a repo-local `.venv`.
- **Data policy**: `.env` or the shell sets a device-specific absolute
  `DATA_DIR` outside the repo. Current cold-run root:
  `/home/tycheng/data/earnings-event-vol`.
- **Task runner**: `just`
- **Core libraries**: `numpy`, `pandas`, `polars`, `pyarrow`, `pydantic`,
  `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `optuna`, `torch`
- **Sequence runtime**: active sequence diagnostics use ridge-flat aggregates
  plus lightweight in-repo PyTorch attention/CNN encoders only. Slow recurrent
  or SSM 5-seed sequence ensembles are not active runtime dependencies.
- **Docs**: `mkdocs` with `mkdocs-material`

# Codebase Structure

- `SPEC.md`: root protocol contract; do not move into docs nav.
- `src/earnings_event_vol/`: active Python package.
- `tests/`: pytest suite with coverage gate.
- `docs/`: reader-facing docs site. Current nav entries are Home, Results
  Snapshot, Paper Plan, Audit Prompts, and Future Work.
- `reports/`: generated report markdown and figures; ignored. Repo-local
  reports are not canonical current evidence.
- `artifacts/`: generated metrics, manifests, and diagnostics; ignored.
  Repo-local artifacts are not canonical current evidence.
- External `DATA_DIR`: generated silver/gold data; ignored by location because
  it is outside the repo.
- `.serena/memories/`: tracked Serena project memory. Use the repo subfolder
  `.serena`, not the parent wrapper `.serena`.
- `.env`: machine-local environment config; ignored.
- `justfile`: standard command surface.
