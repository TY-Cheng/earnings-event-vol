# Audit Status 2026-06-03

Repo root for actual work is `/home/tycheng/projects/earnings-event-vol/earnings-event-vol` inside the outer workspace. `just status` resolves `DATA_DIR=/home/tycheng/data/earnings-event-vol`, artifacts under the inner repo, and Massive secret-file paths configured/existing.

## Current verification

- `just status` passed when run outside sandbox because uv cache is under `/home/tycheng/.cache/uv`.
- `just data args="--dry-run"` passed: active stages are `options-day-aggs-bulk -> universe -> dynamic-calendar -> sec-companyfacts -> event-window-panel -> contract-reference-validation -> trade-proxy-panel`; no data outputs written.
- Targeted validation passed: `python -m pytest` = 120 passed, coverage 95.46%; `python -m mkdocs build --strict --clean` passed.
- Full `just check` was not run during this audit because it auto-formats/lint-fixes and runs source probes.

## Artifact state and caveats

- External data directory is about 4.7G.
- Existing `artifacts/modeling/research_manifest.json` reports `ok=true`, `stage=all`, `sequence_suite=all`, `bootstrap_iter=1000`, `tuning_profile=tuned_phase1`, `feature_schema_version=fe_v2_sec_xbrl`, `tuning_seed=17`.
- Modeling snapshot is still the 2026-05-12 canonical run: feature matrix 810 rows x 532 columns, predictions 2,430 rows x 565 columns, forecast/ranking metrics 48 rows each, strategy metrics 96 rows, 33 trained model-target fits, feature schema report 532 rows / 397 model features.
- Current external `gold/event_panel/trade_proxy_event_panel.parquet` has 816 rows/events, 807 C2C RVAR rows, and 699 `trade_proxy_ivar_event`/`ivar_event` rows. This is newer/different than docs/modeling snapshot that cite 810 events and 693 IVAR rows.
- `artifacts/data_pipeline/data_pipeline_manifest.json` has `ok=false` and a stale blocked `trade-proxy-panel` record with `KeyError: 'eligible_for_quote_pool'`. Current intermediate parquets now contain `eligible_for_quote_pool`, so the manifest likely records an old failed attempt and should be overwritten by a clean/force data run or explicit manifest refresh.
- Research manifest output paths still contain the pre-nested repo root `/home/tycheng/projects/earnings-event-vol/...` instead of the current inner repo root `/home/tycheng/projects/earnings-event-vol/earnings-event-vol/...`, so path portability/stale-artifact checks are needed.

## Research answer

The research question is whether models improve trading decisions around option-implied earnings event variance mispricing, not generic IV forecasting. Market baseline is `IVAR_event`; C2C ex-post mispricing is `RVAR_event_day_c2c - IVAR_event`; premium-space expected edge and proxy PnL are the economic layer.

Targets: `jump_c2o` primary scientific ranking target, `day_c2c` only V1 proxy-PnL headline, `reaction_o2c` diagnostic post-open digestion target.

Models/benchmarks: market IVAR, last-four RVAR, last-four IVAR, Goyal-Saretto RV-IV spread, Elastic Net, LightGBM, XGBoost, LightGBM/XGBoost ensemble, FT-Transformer, ridge-flat sequence, 5-seed BiGRU, official `mamba-ssm` 5-seed, attention pooling, dilated CNN, mask-only, time-shuffle.

Metrics: MAE/RMSE/QLIKE diagnostic/OOS R2 vs IVAR; AUC/top-decile precision/Brier/calibration/edge-decile monotonicity; gross/net proxy PnL, return on premium/capital, Sharpe/Sortino, max drawdown, hit rate, avg win/loss, tail loss, turnover; cost sensitivity, bootstrap/inference diagnostics, sequence drop rate.

Current sell: conservative proxy-stage signal screening. FE V1 LightGBM/XGBoost ablation is the strongest sell; FE V2 default is a negative diagnostic; sequence/Mamba rows are diagnostic only; no paper-grade execution/NBBO claim.

## Docs and legacy

Docs answer the user-facing questions and are close to paper structure: `paper_plan.md` is Abstract -> Intro -> Materials/Methods -> Experiments -> Limitations -> Conclusion; `results_snapshot.md` is a manuscript-like Results and Discussion ledger; `future_work.md` lists paper blockers. Docs are not yet a final submission manuscript because they still need refreshed artifact sync, literature citations, final robustness/inference, and quote/NBBO execution data.

Legacy cleanup is partial. Retired fake Mamba ids remain only as retired manifest/test assertions and are not active models. `fe_v1_legacy` intentionally remains for ablation. Generated artifacts/reports/site are ignored. `.serena/` is untracked because `.gitignore` currently comments out `.serena/`; decide whether to ignore or commit selected memories.

## Recommended next run order

1. Resolve stale artifact ledger: either force-refresh `trade-proxy-panel`/data manifest or run the full data DAG cleanly so `data_pipeline_manifest.json` becomes `ok=true` under the inner repo root.
2. Re-run canonical FE V2 research on the refreshed panel.
3. Re-run FE V1 same-code ablation and regenerate the ablation summary.
4. Run `just research-report`, sync figures, update `docs/results_snapshot.md` from artifacts.
5. Run full `just check` only after accepting that it may format/lint-fix the working tree and run source probes.

## Bottom line

The repo is good enough for an internal working-paper/proxy-stage research review. It is not yet paper-grade/submission-ready. Missing blockers are quote/NBBO or equivalent data, quote-based IVAR, leg-level bid/ask execution, 2013-2025 full sample, DTE/liquidity/regime robustness, stronger inference, and artifact/doc synchronization after the latest panel state.