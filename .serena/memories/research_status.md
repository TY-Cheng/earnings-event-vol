# Research Status

Last synced: 2026-06-13 after switching the main target window to
2016-10-01 through 2026-06-05. The Mac checkout still has a broader
2016-01-01 preflight feature snapshot; the full main-window model/report
package has not yet been rerun.

## Verification State

- Full `just check` passed on 2026-06-13 after the latest
  contract-reference, manifest-canonicalization, rate-limiter, feature,
  docs, and research-helper fixes: 175 tests passed, total coverage 95.13%,
  ruff format/check passed, mypy passed, MkDocs strict build passed, CLI
  `status` passed, and `source-probe all` passed.
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

- Main target rebuild/data window: 2016-10-01 through 2026-06-05.
- Bronze options day aggs are covered in the broader Mac preflight snapshot.
  Bronze underlying day aggs are complete from 2016-06-13 onward, so the main
  2016-10-01 window starts after the known 2016-H1 Massive underlying daily-bar
  entitlement gap.
- The broader 2016-01-01 Mac preflight snapshot has 5,785 in-universe dynamic
  calendar rows and 3,072 proxy-timing main-sample candidates. Its rebuilt
  event-window panel has 3,072 events,
  3,001 events with RVAR, 80,275 candidate contract rows, and 40,709
  quote-pool contract rows.
- Contract-reference validation in the preflight snapshot has been rebuilt:
  79,903 unique option tickers, 79,634 validated, and 269 `missing_reference`;
  unknown deliverables are not proxy-usable.
- Trade-proxy panel in the preflight snapshot has been rebuilt: 3,072 events,
  3,001 with RVAR, 2,538 with
  trade-proxy IVAR, 80,006 proxy-usable contract rows, 55,580 contracts with
  usable pre-entry trade proxy marks, and 24,426 with no trade in the cutoff
  window.
- Gold feature matrix and feature-schema report in the preflight snapshot have
  been rebuilt: 3,071 rows, 559 columns, and 415 model features under
  `fe_v2_sec_xbrl`. Rebuild them for the 2016-10-01 main window before citing
  current-window feature/model claims.
- Model/report outputs still need a main-window rerun. A local
  no-sequence model smoke rerun reached `lightgbm_tuned` and then segfaulted
  in the Mac LightGBM runtime even with single-thread environment variables.
  A separate minimal LightGBM fit succeeds, so the blocker appears tied to the
  current research training path and this Mac runtime combination rather than
  a total LightGBM import failure. Rerun models/report on a stable CPU/GPU
  environment before citing current selected parameters, forecast/ranking
  metrics, strategy rows, or PnL.
- Panel grade remains `no_nbbo_trade_proxy`; `paper_grade=false` for canonical economics.
- Lake-quality audit is implemented and populated at `artifacts/data_pipeline/lake_quality_audit/`.
- Latest preflight lake audit remains `ok=false`: options day aggs are covered,
  but missing full bid/ask/NBBO-equivalent quote coverage keeps the main-window
  package from being paper-grade.

## Quote Execution State

Bounded targeted quote extraction is populated.

- Older normalized targeted quote rows had 10,921,438 rows after bounded shard
  consolidation; these counts predate the current 2016-10-01 main-window
  rebuild.
- Quote-window requests: 14,366 rows, 502 events.
- Quote-window marks: 14,366 rows, 502 events.
- Quote execution legs: 14,366 rows, 502 events.
- Quote straddle execution: 3,599 rows, 502 events.
- Quote-IVAR diagnostic: 502 rows, 502 events, with 468 finite `quote_mid_ivar_event` values.
- Bounded quote-IV surface: 7,183 leg rows with 7,164 finite `quote_mid_iv` values.
- Bounded quote-IV surface summary: 3,599 pair rows with 3,573 finite quote mid-total-variance values.
- Bounded quote-surface IVAR: 502 event rows with 471 finite mid-IVAR rows.
- Execution confidence: 502 rows; 448 high, 53 medium, and 1 low.
- REST quote cache exists under the active `DATA_DIR` in
  `bronze/massive/quotes_v3_rest_target_windows/cache`.
- Targeted REST quote extraction supports `--quote-workers` / `quote_workers` for bounded parallel window fetches while preserving the cache and no-full-day-file policy.
- Targeted quote extraction supports resumable batch slices with `--quote-event-offset`, `--max-events`, and `--quote-batch-label`; batch-labeled runs write under `batches/batch=...` and do not overwrite the canonical bounded quote slice. Use `quote-execution-merge` with `--quote-merge-batch` after shard verification to consolidate batches into canonical quote lake/research artifacts.
- Full-day quote files are not stored in the repo.
- `quote_ivar_event` remains a diagnostic premium-total-variance proxy. The quote-IV surface artifacts are bounded diagnostics, not full target-window NBBO-equivalent surface coverage.

## Historical Model/Report Package

The files below exist from the historical model/report snapshot and should be
regenerated before current target-window claims:

- `quote_confidence_prediction_coverage.csv`: 30 rows with high/medium/low/missing bands across train/validation/test.
- `quote_confidence_strategy_summary.csv`: 72 rows.
- `quote_confidence_ivar_defeat_summary.csv`: 126 rows.
- `quote_confidence_casebook_summary.csv`: 531 rows.
- `quote_ivar_summary.csv`: 4 rows.
- `robustness_summary.csv`: 18 rows covering robustness slices including liquidity/DTE/regime/quote confidence dimensions.
- `feature_schema_report.csv`: refreshed for `fe_v2_sec_xbrl` with 415 model
  features, including sequence call/put volume imbalance aggregates,
  own-underlying pre-event return/RV run-up, and SEC SIC coarse controls.
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
2. If paper-grade execution is required, acquire/fill historical quote/NBBO-equivalent coverage for 2016-10-01 through 2026-06-05; current Massive underlying daily entitlement starts on 2016-06-13, so the active main window avoids the earlier 2016-H1 underlying-label blocker.
3. Keep sequence results framed as failed-gate diagnostics unless future runs
   beat controls and economics gates.
4. Re-run models/report under `tuned_phase1_day_c2c_rank_log_rvar` without
   relying on stale selected params if current-code metrics are needed, then
   run full canonical research after final data and model-scope decisions.
5. Keep `paper_plan.md`, `results_snapshot.md`, README, and Serena memories synchronized with verified artifacts only.
