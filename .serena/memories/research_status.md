# Research Status

Last synced: 2026-06-12 after switching the canonical tuning contract to the
log-target profile and dual-output LightGBM/XGBoost ensemble, without running a
full retune.

## Verification State

- Previous full quality gate: `just check` passed on 2026-06-12 before this
  model cleanup.
- Current cleanup verification ran ruff, targeted model/CLI/report tests, and
  MkDocs strict build. Full `just check` was not rerun after cleanup.
- Active research code no longer accepts `mamba_backend` or `mamba_seeds`,
  no longer exposes `bigru_sequence_5seed` or `mamba_ssm_sequence_5seed`, and
  no longer imports or instantiates a BiGRU sequence encoder.
- Latest populated active-suite model artifacts were refreshed before the
  sequence-control runtime cleanup and before the switch to the canonical
  `tuned_phase1_day_c2c_rank_log_rvar` profile with
  `research --stage models --sequence-suite all --bootstrap-iter 0
  --reuse-tuning-params`: 2,448 prediction rows, 42 forecast rows, 42 ranking
  rows, 84 strategy rows, 15 sequence gate rows, 15 incremental-value rows,
  and `trained_models=27`. The report/completion-gap artifacts still need a
  report-stage refresh, and mask-only/time-shuffle numeric rows need a
  model/report rerun before they represent current code.
- Current code defaults to `tuned_phase1_day_c2c_rank_log_rvar`: Elastic Net,
  LightGBM, XGBoost, and FT-Transformer train on
  `log(max(RVAR, 0) + 1e-6)` and back-transform forecasts to raw variance
  units before metrics and strategy logic. Old `jump_c2o`, raw-target, or
  old-profile selected-parameter caches are rejected.
- Completion-gap audit: `artifacts/modeling/completion_gap_audit.json` has `ok=false`, `paper_grade_ready=false`, 12 audit rows, status counts `complete=8`, `diagnostic_only=2`, `incomplete=2`.
- This is still proxy-stage evidence, not final target-window paper-grade NBBO execution evidence.

## Current Data State

- Proxy/modeling window: 2022-12-01 through 2025-12-31.
- Target rebuild/paper window: 2013-01-01 through 2026-06-05, pending historical quote/NBBO or equivalent data.
- Main feature matrix: 816 BMO/AMC events.
- Panel grade remains `no_nbbo_trade_proxy`; `paper_grade=false` for canonical economics.
- Lake-quality audit is implemented and populated at `artifacts/data_pipeline/lake_quality_audit/`.
- Latest lake audit: `ok=false`; 17/17 audited datasets are target-span incomplete; 15/15 required paper-grade datasets are incomplete for the target window.

## Quote Execution State

Bounded targeted quote extraction is populated.

- Normalized targeted quote rows: 10,921,438 rows at `/home/tycheng/data/earnings-event-vol/bronze/massive/quotes_v1_target_windows/quote_window_quotes.parquet` after canonical plus `offset64_size64`, `offset128_size64`, `offset192_size64`, `offset256_size64`, `offset320_size16`, `offset336_size16`, `offset352_size16`, `offset368_size16`, `offset384_size16`, `offset400_size16`, `offset416_size16`, `offset432_size16`, `offset448_size16`, `offset464_size16`, `offset480_size16`, and `offset496_size16` consolidation.
- Quote-window requests: 14,366 rows, 502 events.
- Quote-window marks: 14,366 rows, 502 events.
- Quote execution legs: 14,366 rows, 502 events.
- Quote straddle execution: 3,599 rows, 502 events.
- Quote-IVAR diagnostic: 502 rows, 502 events, with 468 finite `quote_mid_ivar_event` values.
- Bounded quote-IV surface: 7,183 leg rows with 7,164 finite `quote_mid_iv` values.
- Bounded quote-IV surface summary: 3,599 pair rows with 3,573 finite quote mid-total-variance values.
- Bounded quote-surface IVAR: 502 event rows with 471 finite mid-IVAR rows.
- Execution confidence: 502 rows; 448 high, 53 medium, and 1 low.
- REST quote cache exists under `/home/tycheng/data/earnings-event-vol/bronze/massive/quotes_v3_rest_target_windows/cache`.
- Targeted REST quote extraction supports `--quote-workers` / `quote_workers` for bounded parallel window fetches while preserving the cache and no-full-day-file policy.
- Targeted quote extraction supports resumable batch slices with `--quote-event-offset`, `--max-events`, and `--quote-batch-label`; batch-labeled runs write under `batches/batch=...` and do not overwrite the canonical bounded quote slice. Use `quote-execution-merge` with `--quote-merge-batch` after shard verification to consolidate batches into canonical quote lake/research artifacts.
- Full-day quote files are not stored in the repo.
- `quote_ivar_event` remains a diagnostic premium-total-variance proxy. The quote-IV surface artifacts are bounded diagnostics, not full target-window NBBO-equivalent surface coverage.

## Current Research Package

- `quote_confidence_prediction_coverage.csv`: 30 rows with high/medium/low/missing bands across train/validation/test.
- `quote_confidence_strategy_summary.csv`: 72 rows.
- `quote_confidence_ivar_defeat_summary.csv`: 126 rows.
- `quote_confidence_casebook_summary.csv`: 531 rows.
- `quote_ivar_summary.csv`: 4 rows.
- `robustness_summary.csv`: 18 rows covering robustness slices including liquidity/DTE/regime/quote confidence dimensions.
- `sequence_model_fit_diagnostics.csv`: 15 rows.
- `incremental_value_diagnostics.csv`: 15 rows.

## Model Status

Implemented model families: market IVAR, last-four RVAR, last-four IVAR,
Goyal-Saretto RV-IV spread, Elastic Net, LightGBM, XGBoost,
LightGBM/XGBoost forecast ensemble, FT-Transformer, ridge-flat sequence
aggregates, attention pooling, dilated CNN, mask-only, and time-shuffle
controls. The Goyal-Saretto row is an earnings-event RV-IV spread benchmark,
not a full original-portfolio replication. The active LightGBM/XGBoost
ensemble is dual-output: raw variance forecast average for expected-edge
magnitude and split-percentile base-edge rank average for ranking/top-k
ordering.

Current sequence-suite refresh:

- Active sequence diagnostics are ridge-flat, attention pooling, dilated CNN,
  mask-only, and time-shuffle; the control rows now use the lightweight CNN
  runtime instead of a GRU.
- The removed 5-seed recurrent/SSM sequence ensembles are no longer runnable
  public model ids and no longer have CLI seed/backend knobs.
- Primary `jump_c2o` sequence rows remain diagnostic until they beat
  mask-only/time-shuffle controls, tabular baselines, and economics gates.
- XGBoost tuning now includes depths through 6, lower learning-rate candidates,
  continuous uniform `min_child_weight` over `[3, 50]`, a 0.01 penalty for
  `best_iteration < 25`, `tree_method="hist"`, controlled `n_jobs`, and
  best-trial selection by the penalized objective. The 0.01 penalty is a soft
  discouragement, not a hard veto.
- Current populated numeric snapshot predates the log-target profile and
  forecast-ensemble dual-output code change; rerun models before citing new
  selected params, new ensemble rows, IVAR-defeat rows, or casebook rows.

## Current Sell and Boundaries

The defensible sell remains conservative signal screening with better
benchmark discipline and bounded quote-aware diagnostics. The project can say
it now has targeted quote-execution diagnostics, quote-confidence
stratification, IVAR defeat/casebook summaries, lake-quality gates, and a
lightweight sequence diagnostic suite.

Do not claim paper-grade execution, full bid/ask/NBBO execution, positive
trading outperformance, sequence superiority, complete target-window data
coverage, or a full quote-IV surface.

## Next Run Order

1. Decide whether to expand quote extraction to the full final sample or keep bounded quote diagnostics as a paper limitation; if expanding, use batch-labeled offset/size slices for resumable coverage and consolidate verified shards with `quote-execution-merge`.
2. If paper-grade execution is required, acquire/fill historical quote/NBBO-equivalent coverage for 2013-01-01 through 2026-06-05.
3. Keep sequence results framed as failed-gate diagnostics unless future runs
   beat controls and economics gates.
4. Re-run models/report under `tuned_phase1_day_c2c_rank_log_rvar` without
   relying on stale selected params if current-code metrics are needed, then
   run full canonical research after final data and model-scope decisions.
5. Keep `paper_plan.md`, `results_snapshot.md`, README, and Serena memories synchronized with verified artifacts only.
