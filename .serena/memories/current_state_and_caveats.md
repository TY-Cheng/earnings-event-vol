# Current State and Caveats

Last synchronized: 2026-06-13 after switching the main target window to
2016-10-01 through 2026-06-05. The Mac checkout has a broader 2016-01-01
preflight data/feature snapshot, but the full main-window data/feature/model
package still needs to be rerun.

Current Mac checkout root is `/Users/tycheng/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/earnings-event-vol/earnings-event-vol`; the outer `/Users/tycheng/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/earnings-event-vol` is only the local workspace wrapper. Other machines should use their active checkout path and machine-local `.env`.

## Local Execution Status

- `DATA_DIR`, `UV_PROJECT_ENVIRONMENT`, and secret-file paths are
  machine-local `.env` settings. Do not hardcode the Mac paths in portable
  commands or docs.
- The most recent Mac verification resolved `DATA_DIR` to
  `/Volumes/ExternalSSD/data/earnings-event-vol`; this is an example of the
  local `.env`, not a required project path.
- Current `just check` status: passed on 2026-06-13 with 175 tests, coverage
  95.13%, ruff format/check clean, mypy clean, MkDocs strict build clean, CLI
  `status` clean, and `source-probe all` clean.
- Data command defaults now target 2016-10-01 through 2026-06-05. The broader
  2016-01-01 Mac materialization is a preflight snapshot only after the
  target-window switch.
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

- `day_c2c`: default hyperparameter-selection target and
  literature-compatible target.
- `jump_c2o`: primary scientific decomposition target, close-to-open earnings
  jump variance.
- `reaction_o2c`: post-open digestion diagnostic.

## Current Data and Execution Grade

- Target rebuild/data window: 2016-10-01 through 2026-06-05.
- The broader Mac preflight snapshot shows bronze options day aggregates
  covering the prior broad window. Bronze underlying day aggregates cover
  2016-06-13 through 2026-06-05, so the main window starts after the known
  2016-H1 underlying entitlement gap.
- The broader Mac preflight snapshot has 5,785 in-universe dynamic-calendar
  rows and 3,072 main-sample proxy-timing candidates after SEC primary-document
  text validation.
- The broader preflight event-window panel has 3,072 events, 3,001 events with
  RVAR, 80,275 candidate contract rows, and 40,709 quote-pool contract rows.
- Preflight contract-reference validation covers 79,903 unique option tickers:
  79,634 validated and 269 `missing_reference`; unknown deliverables are not
  proxy-usable, and the preflight candidate panel has 40,643 eligible quote-pool
  rows and 17,577 main-DTE rows after reference gating.
- Trade-proxy panel has been rebuilt for the preflight event-window panel:
  3,072 events, 3,001 with RVAR, 2,538 with trade-proxy IVAR, 80,006
  reference-usable proxy contract rows, 55,580 contracts with usable
  pre-entry trade proxy marks, and 24,426 with no trade in the cutoff window.
- Main second-aggregate cache in the broader preflight refresh has 79,920
  files, with 70,825 written and 9,095 cache hits. Exit-preclose and post-open
  caches each have 5,400 files, with 4,134 written and 1,266 hits.
- Preflight gold feature matrix and `feature_schema_report.csv` have been
  refreshed:
  3,071 feature rows, 559 total columns, and 415 model features under
  `fe_v2_sec_xbrl`. One LCID non-trading-day event without entry/exit timestamp
  is filtered before feature construction.
- Full model/report outputs have not yet been rebuilt against a refreshed
  2016-10-01 main-window feature matrix. A local no-sequence model smoke rerun
  against the broader preflight matrix reached
  `lightgbm_tuned` and then segfaulted in the Mac LightGBM runtime even with
  single-thread environment variables. A separate minimal LightGBM fit succeeds,
  so the blocker appears tied to the current research training path and this
  Mac runtime combination rather than a total LightGBM import failure. Rerun
  models/report on a stable CPU/GPU environment before citing current PnL or
  selected parameters.
- Panel grade remains `no_nbbo_trade_proxy`; `paper_grade=false` for canonical model economics.
- Full historical lake-quality audit is implemented and populated at
  `artifacts/data_pipeline/lake_quality_audit/`. For the main 2016-10-01
  window, the expected remaining paper-grade blocker is full historical
  bid/ask/NBBO-equivalent quote coverage.

## Quote Execution State

Bounded quote extraction below is from the older proxy snapshot unless
explicitly refreshed after the 2016-10-01 main-window event panel. It is not
full-sample paper-grade NBBO evidence.

- Route: targeted Massive REST quote windows with cache under the active
  `DATA_DIR`, usually `bronze/massive/quotes_v3_rest_target_windows/cache`.
- Targeted REST quote extraction supports `--quote-workers` / `quote_workers` for bounded parallel window fetches while preserving the cache and no-full-day-file policy.
- Targeted quote extraction also supports resumable batch slices with `--quote-event-offset`, `--max-events`, and `--quote-batch-label`; batch-labeled runs write under `batches/batch=...` and do not overwrite the canonical bounded quote slice. Use `quote-execution-merge` with `--quote-merge-batch` after shard verification to consolidate batches into canonical quote lake/research artifacts.
- No full-day quote files are stored in the repo.
- Older bronze normalized targeted quotes had 10,921,438 rows after bounded
  shard consolidation; do not treat those counts as refreshed for the current
  2016-10-01 main-window event panel.
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

- Default and only accepted feature schema: `fe_v2_sec_xbrl`.
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
- Current populated model metrics predate the refreshed 2016-10-01
  main-window feature matrix; rerun models before citing selected params,
  ensemble rows, IVAR-defeat rows, casebook rows, or strategy PnL.
- Current diagnostics are 15 sequence gate rows and 15 incremental-value rows.
  Sequence rows remain diagnostic; primary `jump_c2o`
  sequence rows must beat controls and bootstrap gates before any headline
  claim.

## Current Research Artifacts

Fresh analysis artifacts include:

- `feature_schema_report.csv`: refreshed for `fe_v2_sec_xbrl` with 415 model
  features, including sequence call/put volume imbalance aggregates,
  own-underlying pre-event return/RV run-up, and SEC SIC coarse controls.
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

1. Re-run models/report under `tuned_phase1_day_c2c_rank_log_rvar` if
   current-code metrics are needed, then keep sequence diagnostics framed as
   failed-gate diagnostics unless future runs beat controls and economics
   gates.
2. Expand quote extraction from bounded diagnostic slice to the final required sample if quote/NBBO or equivalent coverage is available; use batch-labeled offset/size slices to preserve cache reuse and avoid overwriting canonical artifacts, then consolidate verified shards with `quote-execution-merge`.
3. Expand bounded quote-IV surface diagnostics to full historical/NBBO-equivalent coverage if the paper needs quote-derived IVAR as more than bounded diagnostic evidence.
4. Re-run full canonical research only after final data/quote coverage decisions.
5. Keep docs/results synchronized only from verified artifacts.
6. Re-run `just check` before handoff or commit.

## Caveats

- Trade aggregates are OHLCV trade bars, not quotes, bid/ask, or NBBO.
- C2C proxy PnL is economic screening evidence only.
- C2O and O2C proxy PnL are diagnostic decompositions only.
- Paper-grade claims require historical bid/ask or NBBO-equivalent data, quote-based IVAR/surface, leg-level execution with realistic bid/ask crossing, longer historical sample, liquidity/DTE/regime robustness, and clustered/bootstrap inference on the final sample.
