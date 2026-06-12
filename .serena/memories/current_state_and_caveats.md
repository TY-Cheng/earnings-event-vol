# Current State and Caveats

Last synchronized: 2026-06-12 after switching the canonical tuning contract to
the log-target profile and dual-output LightGBM/XGBoost ensemble, without
running a full retune.

Actual repo root is `/home/tycheng/projects/earnings-event-vol/earnings-event-vol`; the outer `/home/tycheng/projects/earnings-event-vol` is only the workspace wrapper.

## Local Execution Status

- Current local `DATA_DIR` is `/home/tycheng/data/earnings-event-vol`.
- `.env` is machine-local and ignored. It should keep `UV_PROJECT_ENVIRONMENT=/home/tycheng/.venvs/earnings-event-vol` and device-specific absolute data/secret paths outside the repo.
- Previous `just check` status: passed on 2026-06-12 with 153 tests, mypy
  clean, MkDocs strict build clean, doc-figure sync check clean, source-probe
  ok, and coverage 95.03%. Cleanup verification reran ruff, targeted tests,
  and MkDocs strict build; full `just check` was not rerun after cleanup.
- Data command defaults target 2013-01-01 through 2026-06-05, but current populated lake coverage is incomplete for that paper target.
- Active research code no longer writes `mamba_seeds`/`mamba_backend`,
  no longer exposes `bigru_sequence_5seed` or `mamba_ssm_sequence_5seed`, and
  no longer imports or instantiates a BiGRU sequence encoder.
- Latest populated active-suite model artifacts were refreshed before the
  sequence-control runtime cleanup and before the switch to the canonical
  `tuned_phase1_day_c2c_rank_log_rvar` profile with `bootstrap_iter=0`:
  `prediction_rows=2448`, `trained_models=27`, 42 forecast rows, 42 ranking
  rows, 84 strategy rows, 15 sequence gate rows, and 15 incremental-value rows.
  Report and completion-gap artifacts still need a report-stage refresh.
  Mask-only/time-shuffle numeric rows need a model/report rerun before they
  represent current code.
- Current code defaults to `tuned_phase1_day_c2c_rank_log_rvar`: learned
  tabular models and FT-Transformer train on `log(max(RVAR, 0) + 1e-6)` and
  back-transform to raw variance units before metrics and strategy logic. It
  rejects stale `jump_c2o`, raw-target, or old-profile selected-parameter
  caches.
- `artifacts/modeling/completion_gap_audit.json` has `ok=false`, `paper_grade_ready=false`, 12 audit rows, and status counts `complete=8`, `diagnostic_only=2`, `incomplete=2`; it marks bounded quote extraction, quote-confidence summaries, bounded quote-IV surface diagnostics, robustness, sequence-suite population, and the lake-quality audit as complete, but full historical/NBBO quote-IV surface coverage, sequence headline gate, target-window data coverage, and paper-grade bid/ask/NBBO execution remain unresolved.

## Research Question and Scope

The paper-facing question is whether models improve trading decisions around option-implied earnings event variance mispricing. This is not generic implied-volatility forecasting. Forecasting is evaluated alongside ranking quality and premium-space proxy economics after costs.

Target system:

- `day_c2c`: default hyperparameter-selection target, literature-compatible
  target, and the only V1 proxy-PnL headline.
- `jump_c2o`: primary scientific decomposition target, close-to-open earnings
  jump variance.
- `reaction_o2c`: post-open digestion diagnostic.

## Current Data and Execution Grade

- Current proxy/modeling window: 2022-12-01 through 2025-12-31.
- Target rebuild/paper window: 2013-01-01 through 2026-06-05, pending historical quote/NBBO or equivalent data.
- Current feature/event panel: 816 BMO/AMC main-sample events.
- Panel grade remains `no_nbbo_trade_proxy`; `paper_grade=false` for canonical model economics.
- Full historical lake-quality audit is implemented and populated at `artifacts/data_pipeline/lake_quality_audit/`. Latest audit: `ok=false`; 17/17 audited datasets are `target_span_incomplete`; all 15 required paper-grade datasets are incomplete for the target window.
- Options day aggregates cover 2022-05-04 to 2025-12-31. Underlying day aggregates cover 2016-05-04 to 2025-12-31. Main event/modeling sample starts in December 2022.

## Quote Execution State

Bounded quote extraction is populated, but it is not full-sample paper-grade NBBO evidence.

- Route: targeted Massive REST quote windows with cache under `/home/tycheng/data/earnings-event-vol/bronze/massive/quotes_v3_rest_target_windows/cache`.
- Targeted REST quote extraction supports `--quote-workers` / `quote_workers` for bounded parallel window fetches while preserving the cache and no-full-day-file policy.
- Targeted quote extraction also supports resumable batch slices with `--quote-event-offset`, `--max-events`, and `--quote-batch-label`; batch-labeled runs write under `batches/batch=...` and do not overwrite the canonical bounded quote slice. Use `quote-execution-merge` with `--quote-merge-batch` after shard verification to consolidate batches into canonical quote lake/research artifacts.
- No full-day quote files are stored in the repo.
- Bronze normalized targeted quotes: `/home/tycheng/data/earnings-event-vol/bronze/massive/quotes_v1_target_windows/quote_window_quotes.parquet` has 10,921,438 rows after canonical plus `offset64_size64`, `offset128_size64`, `offset192_size64`, `offset256_size64`, `offset320_size16`, `offset336_size16`, `offset352_size16`, `offset368_size16`, `offset384_size16`, `offset400_size16`, `offset416_size16`, `offset432_size16`, `offset448_size16`, `offset464_size16`, `offset480_size16`, and `offset496_size16` consolidation.
- Quote requests: 14,366 rows, 502 events.
- Quote marks: 14,366 rows, 502 events.
- Quote execution legs: 14,366 rows, 502 events.
- Quote straddle execution: 3,599 rows, 502 events.
- Quote-IVAR diagnostic: 502 rows, 502 events, with 468 finite `quote_mid_ivar_event` values.
- Bounded quote-IV surface: 7,183 leg rows with 7,164 finite `quote_mid_iv` values.
- Bounded quote-IV surface summary: 3,599 pair rows with 3,573 finite quote mid-total-variance values.
- Bounded quote-surface IVAR: 502 event rows with 471 finite mid-IVAR rows.
- Execution confidence: 502 rows, 448 high, 53 medium, and 1 low.
- `quote_ivar_event` is a diagnostic premium-total-variance proxy. The quote-IV surface artifacts are bounded diagnostics, not full target-window NBBO-equivalent surface coverage.

## Model and Feature Status

- Default feature schema: `fe_v2_sec_xbrl`.
- Ablation schema retained: `fe_v1_legacy`.
- Model-feature allowlist is `artifacts/modeling/feature_schema_report.csv`.
- Execution-confidence, quote-IVAR, and quote-IV surface diagnostics are evaluation fields, not model features.
- Implemented models: market IVAR, last-four RVAR, last-four IVAR,
  Goyal-Saretto RV-IV spread, Elastic Net, LightGBM, XGBoost,
  LightGBM/XGBoost forecast ensemble, FT-Transformer, ridge-flat sequence
  aggregates, attention pooling, dilated CNN, mask-only, and time-shuffle
  controls. The Goyal-Saretto row is an earnings-event RV-IV spread benchmark,
  not a full original-portfolio replication. The LightGBM/XGBoost ensemble is
  dual-output: raw variance forecast average for expected-edge magnitude and
  split-percentile base-edge rank average for ranking/top-k ordering.
- Active sequence diagnostics are ridge-flat, attention pooling, dilated CNN,
  mask-only, and time-shuffle. The control rows now use the lightweight CNN
  runtime; the removed 5-seed sequence ensembles are no longer public model ids.
- XGBoost tuning now includes depths through 6, lower learning-rate candidates,
  continuous uniform `min_child_weight` over `[3, 50]`, a 0.01 penalty for
  `best_iteration < 25`, `tree_method="hist"`, controlled `n_jobs`, and
  best-trial selection by the penalized objective. The 0.01 penalty is a soft
  discouragement, not a hard veto.
- Current populated numeric snapshot predates the log-target profile and
  forecast-ensemble dual-output code change; rerun models before citing new
  selected params, new ensemble rows, IVAR-defeat rows, or casebook rows.
- Current diagnostics are 15 sequence gate rows and 15 incremental-value rows.
  Sequence rows remain diagnostic; primary `jump_c2o`
  sequence rows must beat controls and bootstrap gates before any headline
  claim.

## Current Research Artifacts

Fresh analysis artifacts include:

- `quote_confidence_prediction_coverage.csv`: 30 rows with high/medium/low/missing bands across train/validation/test.
- `quote_confidence_strategy_summary.csv`: 72 rows.
- `quote_confidence_ivar_defeat_summary.csv`: 126 rows.
- `quote_confidence_casebook_summary.csv`: 531 rows.
- `quote_ivar_summary.csv`: 4 rows.
- `robustness_summary.csv`: 18 rows.
- `sequence_model_fit_diagnostics.csv`: expected 15 active-suite rows after
  the next model/report refresh.
- `incremental_value_diagnostics.csv`: expected 15 active-suite rows after
  the next model/report refresh.

## Current Sell

The defensible near-term claim is conservative: a reproducible signal-screening
study for earnings event-variance mispricing with strong baseline discipline,
bounded quote-aware diagnostics, and lightweight sequence diagnostics. Do not
claim paper-grade executable performance, full bid/ask or NBBO execution,
positive trading outperformance, sequence superiority, or completed
target-window coverage.

## Next Execution Path

1. Expand quote extraction from bounded diagnostic slice to the final required sample if quote/NBBO or equivalent coverage is available; use batch-labeled offset/size slices to preserve cache reuse and avoid overwriting canonical artifacts, then consolidate verified shards with `quote-execution-merge`.
2. Expand bounded quote-IV surface diagnostics to full historical/NBBO-equivalent coverage if the paper needs quote-derived IVAR as more than bounded diagnostic evidence.
3. Re-run models/report under `tuned_phase1_day_c2c_rank_log_rvar` if
   current-code metrics are needed, then keep sequence diagnostics framed as
   failed-gate diagnostics unless future runs beat controls and economics
   gates.
4. Re-run full canonical research only after final data/quote coverage decisions.
5. Keep docs/results synchronized only from verified artifacts.
6. Re-run `just check` before handoff or commit.

## Caveats

- Trade aggregates are OHLCV trade bars, not quotes, bid/ask, or NBBO.
- C2C proxy PnL is economic screening evidence only.
- C2O and O2C proxy PnL are diagnostic decompositions only.
- Paper-grade claims require historical bid/ask or NBBO-equivalent data, quote-based IVAR/surface, leg-level execution with realistic bid/ask crossing, longer historical sample, liquidity/DTE/regime robustness, and clustered/bootstrap inference on the final sample.
