# Current State and Caveats

Updated originally for the canonical FE V2 tuned run and same-code FE V1 versus FE V2 ablation on 2026-05-12. Current actual repo root is `/home/tycheng/projects/earnings-event-vol/earnings-event-vol`; the outer `/home/tycheng/projects/earnings-event-vol` is only the workspace wrapper.

## Local Execution Status

- MCP/tooling status as of 2026-06-10: Serena CLI is `1.5.3` and Codex now starts Serena with `--project /home/tycheng/projects/earnings-event-vol/earnings-event-vol`; Massive MCP is `mcp-massive v0.9.1` via `mcp_massive_from_file`; J-Quants MCP was reinstalled from official JPX/J-Quants GitHub source `git+https://github.com/J-Quants/j-quants-doc-mcp.git` and Codex uses the installed local command `/home/tycheng/.local/bin/j-quants-doc-mcp`. `uv tool upgrade --all` now succeeds and reports `Nothing to upgrade`; `uv tool list` reports `j-quants-doc-mcp v1.0.0`, while the command's own `--version` still prints `0.1.0`.

- Run commands through WSL from the inner repo, for example:
  `wsl -d Ubuntu --cd /home/tycheng/projects/earnings-event-vol/earnings-event-vol -- bash -lc "just check"`.
- `.env` is machine-local and ignored. It should set
  `UV_PROJECT_ENVIRONMENT=/home/tycheng/.venvs/earnings-event-vol` and a
  device-specific absolute `DATA_DIR` outside the repo.
- Current local `DATA_DIR` is `/home/tycheng/data/earnings-event-vol`.
- Current canonical command:
  `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v2_sec_xbrl"`.
- Active `artifacts/modeling/research_manifest.json` reports `ok=true`,
  `stage=all`, `sequence_suite=all`, `bootstrap_iter=1000`,
  `tuning_profile=tuned_phase1`, `feature_schema_version=fe_v2_sec_xbrl`, and
  `tuning_seed=17`.
- The active FE V2 run produced 2,430 prediction rows and 33 trained
  model-target fits. Event-level model features: 243. Tree model features: 397.
- Tuning artifacts exist at `artifacts/modeling/tuning_trials.csv` and
  `artifacts/modeling/tuning_selected_params.json`; both carry
  `feature_schema_version` and `tuning_profile`, and selected params record
  `test_metrics_used_for_selection=false`.
- FE ablation snapshots are saved locally under
  `artifacts/modeling_ablations/fe_v2_sec_xbrl/` and
  `artifacts/modeling_ablations/fe_v1_legacy/`. Active artifacts were restored
  to FE V2 after the FE V1 run.
- `just mamba-doctor` has passed locally under WSL2 with CUDA and official
  `mamba-ssm`; sequence rows remain diagnostic without the bootstrap gate.
- PR #1 was squash-merged as `6ab8cb9`, then its two Chinese contributor docs
  were deleted after the useful roadmap ideas were absorbed into code and
  paper-facing docs. MkDocs now keeps `Results Snapshot` and `Paper Plan` as
  the front-door research docs.
- `just check` passed after the quote-aware implementation: 129 tests, total
  coverage 95.10%, ruff, mypy, MkDocs strict, status, and source-probe all
  passing.

## Research Question and Scope

The paper-facing question is whether models improve trading decisions around
option-implied earnings event variance mispricing. This is not generic implied
volatility forecasting. Forecasting is evaluated alongside ranking quality and
premium-space proxy economics after costs.

## Current Data and Execution Grade

- Current proxy window: 2022-12-01 through 2025-12-31.
- The 2026-05-12 canonical modeling snapshot used 810 BMO/AMC main-sample events, 801 C2C `rvar_event` rows, 693 trade-proxy `IVAR_event` rows, 12,038 proxy contracts, 10,165 usable pre-cutoff proxy contract prices, 10,138 contracts with local IV proxy, and 779 proxy straddle diagnostic rows.
- The newer external gold `trade_proxy_event_panel.parquet` observed during the 2026-06-03 audit has 816 events, 807 C2C RVAR rows, and 699 IVAR rows. Treat docs/modeling artifacts as needing refresh synchronization before the next canonical rerun.
- Panel grade: `no_nbbo_trade_proxy`.
- `paper_grade=false`; no bid/ask, OPRA, or NBBO execution claim.
- Massive `quotes_v1` flat-file objects are visible, but the canonical run has
  not yet built a filtered quote execution panel. Full-day quote files are too
  large for naive artifact storage; use targeted extraction by event date,
  option ticker, and entry/exit windows.

## Target System

- `jump_c2o`: primary scientific target, close-to-open earnings jump variance.
- `day_c2c`: literature-compatible target and the only V1 proxy-PnL headline.
- `reaction_o2c`: post-open digestion diagnostic.

## Model and Feature Status

- Default feature schema is `fe_v2_sec_xbrl`; `fe_v1_legacy` is retained for
  same-code ablation.
- The resolved `feature_schema_report.csv` allowlist controls model features.
  Raw IDs, raw year/month, exit/outcome/PnL fields, and post-event labels are
  excluded from model inputs.
- `tuned_phase1` is the canonical tuned protocol. It uses train and
  locked-validation rows for selection, then refits on train+validation and
  evaluates locked test once.
- Quote-aware execution confidence is implemented as a diagnostic/evaluation
  layer, not as a model input. New fields such as `execution_confidence_score`,
  `execution_confidence_band`, quote route/status, and spread diagnostics are
  excluded from trainable features by default.
- Current model rows include market IVAR, last-four RVAR, last-four IVAR,
  Goyal-Saretto spread, Elastic Net, LightGBM, XGBoost, LightGBM/XGBoost
  rank-average ensemble, FT-Transformer, ridge-flat sequence aggregates,
  BiGRU 5-seed, official `mamba-ssm` 5-seed, attention pooling, dilated CNN,
  mask-only, and time-shuffle.

## Sequence Status

- Active hybrid sequence tensor: `31 x 21`, with 19 prior daily steps plus
  12 entry-day five-minute trade-aggregate proxy bins.
- Sequence eligible events: 678 out of 810.
- Drop rate: 16.3%; `high_sequence_selection_risk=true`.
- Hybrid sequence is not sparse: 682 events have at least eight valid intraday
  bins; median hybrid feature-mask density is 0.7419.
- Legacy fake Mamba ids (`daily_mamba_20step`, `hybrid_mamba_31step`,
  `intraday_only_mamba_12step`, `mask_only_hybrid_mamba`) are retired because
  they were in-repo gated-RNN variants, not official `mamba-ssm`.
- The full sequence suite did not pass the diagnostic gate. Do not sell Mamba
  or any sequence model as the contribution under current evidence.

## Current Result Summary

Active FE V2 canonical run:

- Best `jump_c2o` AUC: Goyal-Saretto spread, about 0.602.
- Best `jump_c2o` OOS R2 versus IVAR: LightGBM, about 0.203.
- Best `day_c2c` headline proxy PnL: ridge-flat sequence aggregates, about
  19,918 USD, but this remains diagnostic because the sequence gate fails.
- Tuned LightGBM/XGBoost FE V2 rows have weak `jump_c2o` AUC and negative
  `day_c2c` headline proxy PnL.

Same-code FE V1 ablation:

- `jump_c2o`: LightGBM has best AUC, about 0.677; XGBoost has best OOS R2
  versus IVAR, about 0.375.
- `day_c2c`: LightGBM leads the headline proxy strategy, about 53,664 USD net
  PnL; XGBoost has best OOS R2 versus IVAR, about 0.574.
- `reaction_o2c`: ridge-flat sequence leads AUC at about 0.799 and XGBoost has
  best OOS R2, about 0.949; O2C remains diagnostic.

Current sell:

- The defensible signal-screening claim is a parsimonious FE V1 tabular signal
  in a no-NBBO proxy sample.
- FE V2 is a negative diagnostic result, not a headline improvement.
- Sequence rows, including official `mamba-ssm`, remain diagnostic.
- Newly implemented but not yet rerun in canonical artifacts: quote execution
  panel, IVAR defeat tables, and false-positive/false-negative casebook.

## Caveats

- Trade aggregates are OHLCV trade bars, not quotes, bid/ask, or NBBO.
- IV and surface language must say close-trade-implied or trade-aggregate
  proxy.
- C2C proxy PnL is economic screening evidence only.
- C2O and O2C proxy PnL are diagnostic decompositions only.
- Paper-grade claims require historical bid/ask or NBBO-equivalent data,
  quote-based IVAR, leg-level execution with bid/ask crossing, and robust
  inference.
- Next execution path should run `quote-execution-panel` on a bounded event/date
  slice first, inspect quote coverage and confidence bands, then rebuild
  research artifacts so `ivar_defeat_*` and `casebook_*` tables reflect the
  canonical prediction set.
