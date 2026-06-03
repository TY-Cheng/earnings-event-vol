# Tech Stack

- **Language**: Python `>=3.11,<3.14`
- **Package manager**: `uv`
- **Environment policy**: `.env` sets `UV_PROJECT_ENVIRONMENT` outside the
  repo, currently `/home/tycheng/.venvs/earnings-event-vol`
- **Data policy**: `.env` sets a device-specific absolute `DATA_DIR` outside
  the repo, currently `/home/tycheng/data/earnings-event-vol`
- **Task runner**: `just`
- **Core libraries**: `numpy`, `pandas`, `polars`, `pyarrow`, `pydantic`,
  `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `optuna`, `torch`
- **Sequence runtime**: official `mamba-ssm` is optional but available locally
  through the CUDA Mamba install recipe
- **Docs**: `mkdocs` with `mkdocs-material`

# Codebase Structure

- `SPEC.md`: root protocol contract; do not move into docs nav.
- `src/earnings_event_vol/`: active Python package.
- `tests/`: pytest suite with coverage gate.
- `docs/`: reader-facing docs site. Current nav entries are Home, Results
  Snapshot, Paper Plan, Audit Prompts, and Future Work.
- `reports/`: generated report markdown and figures; ignored.
- `artifacts/`: generated metrics, manifests, and diagnostics; ignored.
- External `DATA_DIR`: generated silver/gold data; ignored by location because
  it is outside the repo.
- `.serena/memories/`: local Serena project memory; ignored.
- `.env`: machine-local environment config; ignored.
- `justfile`: standard command surface.
