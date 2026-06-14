# Suggested Commands

Run from:

```bash
cd /home/tycheng/projects/earnings-event-vol/earnings-event-vol
```

Use centralized external venv:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PROJECT_ENVIRONMENT=/home/tycheng/.venvs/earnings-event-vol
export DATA_DIR=/home/tycheng/data/earnings-event-vol
export ARTIFACTS_DIR=/home/tycheng/data/earnings-event-vol/artifacts
export REPORTS_DIR=/home/tycheng/data/earnings-event-vol/reports
export PYTHONPATH=src
```

Do not use a repo-local `.venv`.

## Status and Checks

```bash
just status
just check
```

Direct equivalents:

```bash
uv run --env-file .env python -m earnings_event_vol.cli status
uv run --env-file .env python -m earnings_event_vol.cli source-probe all
```

## Data Audit

```bash
just data args="--stage lake-quality-audit --start 2016-10-01 --end 2026-06-05 --force"
```

## Research Refresh

```bash
just research args="--stage features --feature-schema-version fe_v2_sec_xbrl"
just research args="--stage models --feature-schema-version fe_v2_sec_xbrl --reuse-tuning-params"
just research args="--stage report --feature-schema-version fe_v2_sec_xbrl"
just _sync-doc-figures
```

## Quote Shards

Metadata planning:

```bash
just data args="--stage quote-execution-panel --start 2016-10-01 --end 2026-06-05"
```

Targeted quote shard:

```bash
just data args="--stage quote-execution-panel --start 2016-10-01 --end 2026-06-05 --quote-run --quote-allow-all-dates --quote-source rest --quote-workers 8 --quote-event-offset N --max-events M --quote-batch-label LABEL"
```

Merge verified shards:

```bash
just data args="--stage quote-execution-merge --quote-merge-exclude-canonical --force"
```

## Current Verified Root

Current canonical external artifacts:

```text
/home/tycheng/data/earnings-event-vol
```

Key files:

- `artifacts/data_pipeline/quote_execution_panel/quote_execution_report.json`
- `artifacts/data_pipeline/lake_quality_audit/lake_quality_report.json`
- `artifacts/modeling/feature_matrix_manifest.json`
- `artifacts/modeling/research_manifest.json`
- `artifacts/modeling/research_report_manifest.json`
- `artifacts/modeling/completion_gap_audit.json`
- `artifacts/modeling/forecast_metrics.csv`
- `artifacts/modeling/ranking_metrics.csv`
- `artifacts/modeling/strategy_metrics.csv`
