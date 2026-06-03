# Suggested Commands

The project uses `just` as the command runner and `uv` for package management.
`UV_PROJECT_ENVIRONMENT` should point outside the repo, currently
`/home/tycheng/.venvs/earnings-event-vol`.

Run commands through WSL from Windows/Codex desktop when needed:

```bash
wsl -d Ubuntu --cd /home/tycheng/projects/earnings-event-vol -- bash -lc "just check"
```

## Primary Commands

- `just status`: Lightweight environment diagnostic for resolved local paths
  and source/secret-file configuration; it does not rebuild data or research.
- `just check`: Full handoff gate after code/doc changes. It runs
  `uv sync --all-extras --dev`, `ruff format`, `ruff check --fix`, mypy,
  pytest, MkDocs strict build, CLI status, and `source-probe all`.
- `just data args="--dry-run"`: Dry-run the active `all` data DAG.
- `just data`: Runs the active `all` data route for the current
  2022-12-01 to 2025-12-31 proxy window.
- `just research`: Runs the development-default research package.
- `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v2_sec_xbrl"`:
  Runs the current paper-facing canonical FE V2 tuned package.
- `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v1_legacy"`:
  Runs the same-code FE V1 feature-schema ablation.
- `just research-report`: Regenerates the report and figures from existing
  modeling artifacts.
- `just docs`: Builds and serves the MkDocs site locally.

## Active Data DAG

`just data` runs the active `all` route:

```text
options-day-aggs-bulk -> universe -> dynamic-calendar -> sec-companyfacts
  -> event-window-panel -> contract-reference-validation -> trade-proxy-panel
```

Current defaults include jobs=4, lookback seconds=900, second-aggregate
buffer=60 minutes, price field=`option_vwap`, DTE 3-21, universe top N=50, and
trailing universe lookback=6 months.

## Under-the-Hood CLI

Use these only when a focused lower-level call is necessary:

```bash
PYTHONPATH=src uv run --env-file .env python -m earnings_event_vol.cli status
PYTHONPATH=src uv run --env-file .env python -m earnings_event_vol.cli source-probe all
PYTHONPATH=src uv run --env-file .env python -m earnings_event_vol.cli data --stage <stage> [args]
PYTHONPATH=src uv run --env-file .env python -m earnings_event_vol.cli research [args]
```

## Development Shortcuts

Prefer `just check` before handoff, but targeted checks are:

```bash
uv run ruff format src/earnings_event_vol tests scripts
uv run ruff check --fix src/earnings_event_vol tests scripts
uv run mypy src/earnings_event_vol tests
uv run pytest
uv run mkdocs build --strict --clean
```

## Output Locations

- Data manifest: `artifacts/data_pipeline/data_pipeline_manifest.json`
- Trade-proxy report:
  `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json`
- Feature matrix: `$GOLD_DATA_DIR/modeling/feature_matrix.parquet`
- Hybrid sequence tensor: `$GOLD_DATA_DIR/modeling/hybrid_sequence_tensor_v2.npz`
- Forecast metrics: `artifacts/modeling/forecast_metrics.csv`
- Ranking metrics: `artifacts/modeling/ranking_metrics.csv`
- Strategy metrics: `artifacts/modeling/strategy_metrics.csv`
- Model fit diagnostics: `artifacts/modeling/model_fit_diagnostics.csv`
- Tuning trials: `artifacts/modeling/tuning_trials.csv`
- Tuning selected params: `artifacts/modeling/tuning_selected_params.json`
- Feature-schema ablation snapshots: `artifacts/modeling_ablations/`
- Proxy report: `reports/modeling/proxy_research_report.md`
- Synced docs figures: `docs/assets/images/modeling/`
- Curated reader-facing results: `docs/results_snapshot.md`
