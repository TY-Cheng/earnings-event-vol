# Audit Status

Last synchronized: 2026-06-13 after the WSL2/CUDA cold run.

## Handoff Gate

Latest full gate:

- `just check` must be rerun after the latest quote-shard promotion,
  source-coverage audit, research refresh, docs sync, and legacy cleanup.

## Current Evidence Root

Use:

```text
/home/tycheng/data/earnings-event-vol
```

Do not use repo-local `artifacts/` or `reports/` as current evidence after
cleanup.

## Current Audit Facts

Lake quality:

- `ok=false`
- target window: 2016-10-01 to 2026-06-05
- required datasets: 15
- incomplete required datasets: 13
- `paper_grade_execution_ready=false`

Completion gap:

- `ok=false`
- `paper_grade_ready=false`
- status counts: `complete=8`, `diagnostic_only=1`, `incomplete=3`
- blockers: quote-IVAR/surface paper-grade upgrade, sequence full-suite
  population, target-window data coverage, paper-grade bid/ask/NBBO execution.

## Stale-State Risks

Treat these as stale unless explicitly restored from the cold-run root:

- older 502-event quote slice;
- older 10,921,438 matched quote-row bounded slice;
- older 816-row modeling snapshot;
- broader preflight rows;
- repo-local ignored `artifacts/` and `reports/`;
- parent wrapper `.serena`;
- repo-local `.venv`.

## Current Completion Assessment

Complete for proxy-stage research:

- data engineering through quote batch merge;
- feature matrix;
- model suite;
- report figures;
- quote-confidence summaries;
- IVAR-defeat artifacts;
- casebook artifacts;
- tests/docs build.

Incomplete for paper-grade claims:

- full-window quote/NBBO-equivalent execution;
- full paper-grade quote-IVAR/surface route;
- robust positive economics;
- sequence full-suite model rows in the verified refresh.
