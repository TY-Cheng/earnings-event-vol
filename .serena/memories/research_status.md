# Research Status

Last synced: 2026-06-11 after 64-event bounded quote extraction, bounded quote-IV surface diagnostics, completion-gap audit, quote REST worker and batch-slice wiring, lake-quality audit, explicit GPU-enabled 5-seed sequence/Mamba modeling refresh, docs refresh, and `just check`.

## Verification State

- Full quality gate: `just check` passed on 2026-06-11.
- `just check` evidence: ruff format/check passed, mypy passed across 30 source/test files, 149 pytest tests passed, coverage 95.01%, MkDocs strict build passed, doc-figure sync check passed, source-probe ok.
- Research manifest: `artifacts/modeling/research_manifest.json` has `ok=true`, `stage=models`, `sequence_suite=all`, `mamba_seeds=17,42,123,456,789`, `bootstrap_iter=200`, `tuning_profile=tuned_phase1`, `feature_schema_version=fe_v2_sec_xbrl`, and `reuse_tuning_params=true`.
- Report manifest: `artifacts/modeling/research_report_manifest.json` has `ok=true`, `stage=report`, `sequence_suite=all`, and the same 5-seed Mamba seed list.
- Completion-gap audit: `artifacts/modeling/completion_gap_audit.json` has `ok=false`, `paper_grade_ready=false`, 12 audit rows, status counts `complete=8`, `diagnostic_only=2`, `incomplete=2`.
- This is still proxy-stage evidence, not final 2013-2025 paper-grade NBBO execution evidence.

## Current Data State

- Proxy/modeling window: 2022-12-01 through 2025-12-31.
- Target paper window: 2013-2025, pending historical quote/NBBO or equivalent data.
- Main feature matrix: 816 BMO/AMC events.
- Panel grade remains `no_nbbo_trade_proxy`; `paper_grade=false` for canonical economics.
- Lake-quality audit is implemented and populated at `artifacts/data_pipeline/lake_quality_audit/`.
- Latest lake audit: `ok=false`; 17/17 audited datasets are target-span incomplete; 15/15 required paper-grade datasets are incomplete for the 2013-2025 target.

## Quote Execution State

Bounded targeted quote extraction is populated.

- Normalized targeted quote rows: 1,226,559 rows at `/home/tycheng/data/earnings-event-vol/bronze/massive/quotes_v1_target_windows/quote_window_quotes.parquet`.
- Quote-window requests: 1,642 rows, 64 events.
- Quote-window marks: 1,642 rows, 64 events.
- Quote execution legs: 1,642 rows, 64 events.
- Quote straddle execution: 412 rows, 64 events.
- Quote-IVAR diagnostic: 64 rows, 64 events, with 56 finite `quote_mid_ivar_event` values.
- Bounded quote-IV surface: 821 leg rows with 821 finite `quote_mid_iv` values.
- Bounded quote-IV surface summary: 412 pair rows with 412 finite quote total-variance values.
- Bounded quote-surface IVAR: 64 event rows with 57 finite mid-IVAR rows.
- Execution confidence: 64 rows; 55 high and 9 medium.
- REST quote cache exists under `/home/tycheng/data/earnings-event-vol/bronze/massive/quotes_v3_rest_target_windows/cache`.
- Targeted REST quote extraction supports `--quote-workers` / `quote_workers` for bounded parallel window fetches while preserving the cache and no-full-day-file policy.
- Targeted quote extraction supports resumable batch slices with `--quote-event-offset`, `--max-events`, and `--quote-batch-label`; batch-labeled runs write under `batches/batch=...` and do not overwrite the canonical 64-event quote slice. Use `quote-execution-merge` with `--quote-merge-batch` after shard verification to consolidate batches into canonical quote lake/research artifacts.
- Full-day quote files are not stored in the repo.
- `quote_ivar_event` remains a diagnostic premium-total-variance proxy. The quote-IV surface artifacts are bounded diagnostics, not full 2013-2025 NBBO-equivalent surface coverage.

## Current Research Package

- `quote_confidence_prediction_coverage.csv`: 27 rows with high/medium/missing bands across train/validation/test.
- `quote_confidence_strategy_summary.csv`: 129 rows.
- `quote_confidence_ivar_defeat_summary.csv`: 96 rows.
- `quote_confidence_casebook_summary.csv`: 316 rows.
- `quote_ivar_summary.csv`: 3 rows.
- `robustness_summary.csv`: 18 rows covering robustness slices including liquidity/DTE/regime/quote confidence dimensions.
- `sequence_model_fit_diagnostics.csv`: 21 rows.
- `incremental_value_diagnostics.csv`: 21 rows.

## Model Status

Implemented model families: market IVAR, last-four RVAR, last-four IVAR, Goyal-Saretto RV-IV spread, Elastic Net, LightGBM, XGBoost, LightGBM/XGBoost ensemble, FT-Transformer, ridge-flat sequence aggregates, BiGRU, official `mamba-ssm`, attention pooling, dilated CNN, mask-only, and time-shuffle controls.

Current sequence-suite refresh:

- Explicit ensemble seeds are `17,42,123,456,789`.
- BiGRU 5-seed trained for all 3 targets with `trained_seed_count=5` and 101 locked-test sequence predictions per target.
- Official `mamba-ssm` 5-seed trained for all 3 targets with `trained_seed_count=5`, device `cuda`, and 101 locked-test sequence predictions per target.
- Primary `jump_c2o` sequence gate still fails: Mamba 5-seed has gate AUC lift about 0.0024 with 95% bootstrap CI [-0.0582, 0.0833], so no sequence/Mamba superiority claim.

## Current Sell and Boundaries

The defensible sell remains conservative signal screening with better benchmark discipline and bounded quote-aware diagnostics. The project can say it now has targeted quote-execution diagnostics, quote-confidence stratification, IVAR defeat/casebook summaries, lake-quality gates, and an explicit 5-seed sequence suite including official Mamba.

Do not claim paper-grade execution, full bid/ask/NBBO execution, positive trading outperformance, sequence/Mamba superiority, complete 2013-2025 data coverage, or a full quote-IV surface.

## Next Run Order

1. Decide whether to expand quote extraction to the full final sample or keep bounded quote diagnostics as a paper limitation; if expanding, use batch-labeled offset/size slices for resumable coverage and consolidate verified shards with `quote-execution-merge`.
2. If paper-grade execution is required, acquire/fill historical quote/NBBO-equivalent coverage for 2013-2025.
3. If sequence models remain paper-facing, keep Mamba/BiGRU results framed as failed-gate diagnostics unless future runs beat controls and economics gates.
4. Re-run full canonical research after final data and model-scope decisions.
5. Keep `paper_plan.md`, `results_snapshot.md`, README, and Serena memories synchronized with verified artifacts only.
