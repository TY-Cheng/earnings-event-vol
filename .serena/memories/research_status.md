# Research Status

Last synchronized: 2026-06-13 after the WSL2/CUDA cold-run data engineering
and research refresh.

## Verified Run

Canonical cold-run root:

```text
/home/tycheng/data/earnings-event-vol
```

Window:

```text
2016-10-01 through 2026-06-05
```

Verification:

- `research_manifest.json`: `stage=models`, `ok=true`
- `research_report_manifest.json`: `stage=report`, `ok=true`
- `quote_execution_report.json`: `ok=true`
- `just check`: must be rerun after the latest quote promotion, docs sync, and
  source coverage audit.

## Data State

Feature/model inputs:

- feature matrix rows: 2,388
- feature-schema rows: 569
- model features: 407
- event-level model features: 249
- tree model features: 407

Quote execution:

- route: `quote_batch_consolidation`
- quote-confidence events: 2,329
- quote-window requests: 65,172
- matched quote rows: 21,680,332
- event windows without quote confidence: 60
- targeted request events with zero returned quote rows: 923
- full-day quote files written: false

Lake-quality audit:

- `ok=false`
- required datasets: 15
- incomplete required datasets: 13
- `paper_grade_execution_ready=false`
- blocker: full-window quote/NBBO or equivalent coverage and quote-IVAR beyond
  current diagnostic coverage remain unproven.

Completion-gap audit:

- `ok=false`
- `paper_grade_ready=false`
- status counts: `complete=8`, `diagnostic_only=1`, `incomplete=3`
- blockers: quote-IVAR/surface coverage, sequence full-suite population,
  target-window data coverage, paper-grade bid/ask/NBBO execution.

## Model State

Current profile:

- `tuned_phase1_day_c2c_rank_log_rvar`
- `forecast_floor=1e-6`
- learned tabular models and FT-Transformer train in log-RVAR space and
  evaluate in raw variance space.

Model outputs:

- 27 model-target evaluations
- 7,164 prediction rows
- 27 forecast rows
- 27 ranking rows
- 54 strategy rows
- 27 IVAR-defeat metric rows
- 126 casebook summary rows
- 36 quote-confidence prediction-coverage rows

Implemented active model ids:

- `market_implied_event_variance`
- `last_four_rvar`
- `last_four_ivar`
- `goyal_saretto_rv_iv_spread`
- `linear_elastic_net_tuned`
- `lightgbm_tuned`
- `xgboost_tuned`
- `lightgbm_xgboost_forecast_ensemble`
- `ft_transformer`
Sequence tensors and quality reports are built, but this verified refresh used
`sequence_suite=none`, so sequence model ids are not active metric rows.

Retired/not active:

- 5-seed BiGRU sequence ensemble
- 5-seed Mamba/SSM sequence ensemble
- old Mamba-style model ids

## Current Results

Best ranking AUC:

- `day_c2c`: `lightgbm_xgboost_forecast_ensemble`, AUC 0.5823, top-decile precision 0.4483.
- `jump_c2o`: `goyal_saretto_rv_iv_spread`, AUC 0.5091, top-decile precision 0.2069.
- `reaction_o2c`: `ft_transformer`, AUC 0.6714, top-decile precision 0.2414.

Best forecast OOS R2 versus IVAR:

- `day_c2c`: `xgboost_tuned`, 0.6438.
- `jump_c2o`: `xgboost_tuned`, 0.7301.
- `reaction_o2c`: `linear_elastic_net_tuned`, 0.9312.

Best headline strategy rows:

- `day_c2c`: `xgboost_tuned`, 3 trades, +1,484.41 USD net, 0.4169 return on premium.
- `jump_c2o`: `goyal_saretto_rv_iv_spread`, 5 trades, +3,089.51 USD net, 0.2044 return on premium.
- `reaction_o2c`: `goyal_saretto_rv_iv_spread`, 4 trades, +1,637.39 USD net, 0.2917 return on premium.

Interpretation:

- positive rows are too small-trade for a robust economic alpha claim;
- current sell is signal screening, benchmark discipline, quote-aware
  diagnostics, IVAR-defeat analysis, and casebook interpretation;
- paper-grade execution is blocked.

## Next Research Work

1. Decide whether the paper stops at conservative proxy-stage signal screening
   or invests in full-window quote/NBBO-equivalent execution.
2. If investing in paper-grade execution, fill full historical bid/ask/NBBO or
   equivalent coverage and rerun data/features/models/report.
3. Keep sequence model claims deferred until a deliberate sequence-suite run is
   completed and passes headline gates.
4. Keep docs/README/Serena synchronized from verified artifacts only.
