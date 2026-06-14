# Current State and Caveats

Last synchronized: 2026-06-13 after the WSL2/CUDA cold run.

## Local Path Policy

- Authoritative repo root:
  `/home/tycheng/projects/earnings-event-vol/earnings-event-vol`.
- Do not activate the parent wrapper as the Serena project.
- Parent `/home/tycheng/projects/earnings-event-vol/.serena` is a confusion
  source and should be removed.
- Use external centralized venv:
  `/home/tycheng/.venvs/earnings-event-vol`.
- Do not keep a repo-local `.venv`.
- Canonical current evidence root:
  `/home/tycheng/data/earnings-event-vol`.
- Repo-local `artifacts/` and `reports/` are ignored generated outputs and are
  not canonical current evidence.

## Evidence Grade

Current grade:

- `no_nbbo_trade_proxy`
- proxy-stage signal screening
- quote-aware diagnostics
- not paper-grade executable trading

Do not claim:

- full bid/ask, OPRA, or NBBO execution;
- paper-grade executable strategy performance;
- full-window quote-IVAR/surface coverage;
- robust positive PnL from small-trade rows;
- sequence-model superiority.

## Verified Cold-Run Facts

Window: 2016-10-01 through 2026-06-05.

Quote:

- quote-confidence events: 2,329
- quote-window requests: 65,172
- matched quote rows: 21,680,332
- event windows without quote confidence: 60
- targeted request events with zero returned quote rows: 923
- route: `quote_batch_consolidation`
- full-day quote files written: false

Features:

- feature matrix rows: 2,388
- feature-schema rows: 569
- model features: 407
- event model features: 249
- tree model features: 407

Models/report:

- profile: `tuned_phase1_day_c2c_rank_log_rvar`
- model-target evaluations: 27
- prediction rows: 7,164
- forecast rows: 27
- ranking rows: 27
- strategy rows: 54
- report figures: 11

## Blocking Audits

Lake-quality audit:

- `ok=false`
- required datasets: 15
- incomplete required datasets: 13
- `paper_grade_execution_ready=false`

Completion-gap audit:

- `ok=false`
- `paper_grade_ready=false`
- complete: 8
- diagnostic-only: 1
- incomplete: 3

Blockers:

- quote-IVAR/surface coverage is still diagnostic rather than paper-grade;
- sequence full-suite model rows are not populated in this verified refresh;
- target-window paper-grade data coverage is not proven;
- paper-grade bid/ask/NBBO execution is not complete.

## Current Sell

Allowed sell:

- reproducible earnings event-variance mispricing signal-screening package;
- strong market IVAR, historical, and Goyal-Saretto-style benchmarks;
- log-target tabular/deep model profile;
- quote-aware execution-confidence diagnostics;
- IVAR-defeat and casebook failure analysis.

Forbidden sell:

- executable alpha;
- full spread/NBBO tradability;
- robust positive PnL;
- sequence model contribution as headline.

## Cleanup State

Active code has retired the slow 5-seed BiGRU/Mamba sequence ensembles.
Remaining Mamba/BiGRU mentions should be retired-id manifests or tests proving
they are not active.

Generated repo-local `artifacts/`, `reports/`, and `.venv` should be absent
after cleanup. Current evidence lives outside the repo in the cold-run data
root.
