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

Legacy cleanup is partial. Retired fake Mamba ids remain only as retired manifest/test assertions and are not active models. `fe_v1_legacy` intentionally remains for ablation. Generated artifacts/reports/site are ignored. `.serena/` is tracked project memory/config and currently has local updates; decide whether to commit selected memories with the implementation changes.

## Recommended next run order

1. Resolve stale artifact ledger: either force-refresh `trade-proxy-panel`/data manifest or run the full data DAG cleanly so `data_pipeline_manifest.json` becomes `ok=true` under the inner repo root.
2. Re-run canonical FE V2 research on the refreshed panel.
3. Re-run FE V1 same-code ablation and regenerate the ablation summary.
4. Run `just research-report`, sync figures, update `docs/results_snapshot.md` from artifacts.
5. Run full `just check` only after accepting that it may format/lint-fix the working tree and run source probes.

## Bottom line

The repo is good enough for an internal working-paper/proxy-stage research review. It is not yet paper-grade/submission-ready. Missing blockers are quote/NBBO or equivalent data, quote-based IVAR, leg-level bid/ask execution, 2013-2025 full sample, DTE/liquidity/regime robustness, stronger inference, and artifact/doc synchronization after the latest panel state.

## Update 2026-06-10 MCP/tooling sync

- Actual repo root remains `/home/tycheng/projects/earnings-event-vol/earnings-event-vol`; Codex Serena MCP config was updated to start Serena with this inner repo as `--project` instead of the outer workspace folder.
- Serena CLI/MCP tool is now `Serena 1.5.3`, matching the latest GitHub release observed on 2026-06-10. Activating the inner repo lists the project memories correctly.
- Serena 1.5.x refreshed `.serena/project.yml` with its newer config-template comments/fields; this is a tracked project-config change, not business-code churn.
- Massive MCP remains installed as `mcp-massive v0.9.1`; `uv tool upgrade mcp-massive` reported `Nothing to upgrade`. Codex still launches it through `/home/tycheng/.local/bin/mcp_massive_from_file`, which reads the local Massive API key file and then execs `mcp_massive`.
- J-Quants MCP was reinstalled from the official JPX/J-Quants GitHub source with `uv tool install --reinstall git+https://github.com/J-Quants/j-quants-doc-mcp.git`. Codex config uses the installed local command `/home/tycheng/.local/bin/j-quants-doc-mcp` directly, which is safer here than `uvx j-quants-doc-mcp` because the package was not resolvable from the public registry in this environment.
- After the official J-Quants reinstall, `uv tool upgrade --all` succeeds and reports `Nothing to upgrade`. `uv tool list` reports `j-quants-doc-mcp v1.0.0`; the command's own `--version` still prints `0.1.0`, which appears to be an upstream CLI version-string mismatch.
- Codex project trust config now includes both the outer workspace and the inner repo path.
- Current git working tree after the MCP sync shows `.serena/project.yml` modified. No source-code diff was present in `src/earnings_event_vol/research.py` at the time of this update.

## Update 2026-06-10 PR #1 and quote-aware implementation

- PR #1 was squash-merged into `main` as `6ab8cb9` and then pulled locally. The two Chinese contributor docs from that PR were later deleted because their useful ideas were absorbed into code plus `paper_plan.md` / `results_snapshot.md`; MkDocs nav is back to the paper-facing front door.
- New implementation added `src/earnings_event_vol/quote_execution.py` and CLI command `quote-execution-panel`. It builds targeted quote-window requests, filters Massive `quotes_v1`-like rows by event date / option ticker / entry and exit windows, writes quote-window marks, and builds event-level `execution_confidence_score` / `execution_confidence_band`. It does not store full-day raw quote files.
- Research metrics now emit IVAR defeat and casebook artifacts from locked-test predictions: `ivar_defeat_events.csv`, `ivar_defeat_metrics.csv`, `ivar_defeat_breakdowns.csv`, `casebook_events.csv`, and `casebook_summary.csv`.
- Feature schema now excludes execution-confidence and quote-diagnostic fields from trainable model features by default.
- Docs now state that Massive quote flat-file objects are available but not yet incorporated into the canonical run; targeted extraction and quote-aware artifacts are implemented but the canonical FE V2/FE V1 results have not been rerun.
- Verification after this implementation: `just check` passed with 129 tests, coverage 95.10%, ruff, mypy, MkDocs strict, status, and source-probe all passing. Existing sklearn `n_alphas` deprecation warnings remain non-blocking.
