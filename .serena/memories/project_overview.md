# Project Overview

**Name**: earnings-event-vol

**Repo root**:
Use the active checkout path on the current machine. The last Mac checkout was
`/Users/tycheng/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/earnings-event-vol/earnings-event-vol`.

**Purpose**: reproducible empirical-research pipeline for U.S. single-name
equity-option earnings event variance mispricing.

**Core research question**: Can models improve trading decisions around
option-implied earnings event variance mispricing? This is not generic implied
volatility forecasting.

## Research Contract

- Treat `SPEC.md` as the implementation and research-protocol contract.
- Models forecast realized earnings event variance labels.
- Market baseline: `IVAR_event`.
- Ex post mispricing: `RVAR_event - IVAR_event`.
- Trade decisions use premium-space expected edge, not raw variance edge.
- Current evidence grade is `no_nbbo_trade_proxy` and `paper_grade=false`.

## Target System

- `day_c2c`: default hyperparameter-selection target and
  literature-compatible target.
- `jump_c2o`: primary scientific decomposition target, close-to-open earnings
  jump variance.
- `reaction_o2c`: post-open digestion diagnostic.

## Current Data State

- Target rebuild/data window: 2016-10-01 through 2026-06-05.
- Dynamic monthly top-50 liquid U.S. single-name option underlyings.
- SEC EDGAR 8-K Item 2.02 discovery with SEC primary-document text validation.
- SEC CompanyFacts XBRL fundamentals with conservative as-of gating.
- Massive options day aggregates, underlying day aggregates, targeted option
  one-second trade aggregates, and targeted `quotes_v1` extraction support.
- The Mac checkout currently has a broader 2016-01-01 preflight event-window
  panel: 3,072 events, 3,001 events with RVAR,
  80,275 event contract candidates, and 40,709 quote-pool contract rows.
- Contract-reference validation and trade-proxy panel have been rebuilt for
  this preflight panel. Its trade-proxy panel has 3,072 events, 2,538
  events with trade-proxy IVAR, 80,006 proxy-usable contract rows, and 55,580
  contracts with usable pre-entry trade proxy marks.
- Gold feature matrix in the preflight snapshot has 3,071 rows, 559 columns,
  and 415 model features under `fe_v2_sec_xbrl`. Rebuild data/features for the
  2016-10-01 main window before citing current-window model claims.

## Quote-Aware Evidence Modules

- `quote-execution-panel` builds targeted quote-window requests from event
  windows and candidate contracts.
- It filters local or Massive `quotes_v1` rows by event date, option ticker,
  and entry/exit windows.
- It avoids storing full-day raw quote files in the repo.
- Current bounded real quote run is populated and consolidated through
  `offset64_size64`, `offset128_size64`, `offset192_size64`,
  `offset256_size64`, `offset320_size16`, `offset336_size16`,
  `offset352_size16`, `offset368_size16`, `offset384_size16`,
  `offset400_size16`, `offset416_size16`, `offset432_size16`,
  `offset448_size16`, `offset464_size16`, `offset480_size16`, and
  `offset496_size16`: 14,366 quote-window requests, 10,921,438 matched quote
  rows, 14,366 quote marks, 14,366 leg execution rows, 3,599 straddle execution
  rows, 502 quote-IVAR diagnostic rows, 7,183 bounded quote-IV leg rows, 3,599
  bounded quote-IV surface-pair rows, 502 bounded surface-IVAR event rows, and
  502 execution confidence rows.
- The bounded quote-IV surface has 7,164 finite `quote_mid_iv` values, 3,573
  finite quote mid-total-variance rows, and 471 finite surface-IVAR mid rows.
- Outputs include `quote_window_requests.csv`, `quote_window_quotes.csv`,
  `quote_window_marks.csv`, `quote_execution_legs.csv`,
  `quote_straddle_execution.csv`, `quote_ivar_event.csv`,
  `quote_iv_surface.csv`, `quote_iv_surface_summary.csv`,
  `quote_surface_ivar_event.csv`, `quote_execution_confidence.csv`, and
  `quote_execution_report.json`.
- `quote_ivar_event` is a diagnostic premium-total-variance proxy. The
  quote-IV surface artifacts are bounded diagnostics, not full historical
  NBBO-equivalent surface coverage.

## Feature Protocol

- Default and only accepted schema: `fe_v2_sec_xbrl`.
- Resolved allowlist: `artifacts/modeling/feature_schema_report.csv`.
- Transform artifact: `artifacts/modeling/feature_transform_params.json`.
- Execution confidence, quote diagnostics, quote-IVAR, and quote-IV surface fields are
  evaluation/casebook fields, not trainable model features.

## Implemented Models

Benchmarks:

- market-implied IVAR baseline
- last-four RVAR
- last-four IVAR
- Goyal-Saretto-style RV-IV spread

Tabular and deep models:

- Elastic Net via sklearn `ElasticNetCV`
- LightGBM
- XGBoost
- LightGBM/XGBoost dual-output ensemble: raw forecast average for edge
  magnitude plus split-percentile base-edge rank score for ranking/top-k
  ordering
- FT-Transformer

Sequence diagnostics:

- ridge-flat sequence aggregates
- attention pooling
- non-causal dilated CNN
- mask-only and deterministic time-shuffle controls

The active sequence runtime is lightweight only: ridge-flat aggregates plus
in-repo attention/CNN encoders. Slow recurrent/SSM 5-seed sequence ensembles
are not active model ids or runtime dependencies.

Retired legacy ids:

- `daily_mamba_20step`
- `hybrid_mamba_31step`
- `intraday_only_mamba_12step`
- `mask_only_hybrid_mamba`

These legacy ids were retired before the current lightweight sequence suite and
remain historical only.

## Current Result Summary

Latest synchronized run:

- Intended current command shape: main-window data rebuild through
  contract-reference validation and trade-proxy panel, followed by
  `research --stage features --feature-schema-version fe_v2_sec_xbrl` and
  `data --stage lake-quality-audit`.
- Broader preflight manifest: `ok=true`, `stage=features`,
  `feature_schema_version=fe_v2_sec_xbrl`, `sequence_suite=all`,
  `tuning_profile=tuned_phase1_day_c2c_rank_log_rvar`.
- Broader preflight event-window rows: 3,072.
- Broader preflight feature matrix rows: 3,071.
- Broader preflight feature matrix columns: 559, with 415 model features.
- Historical model/report snapshot rows: 816 feature rows and 2,448
  prediction rows. Those forecast/ranking/strategy metrics predate the
  refreshed feature matrix and are stale for current-code claims.
- Current code rejects stale `jump_c2o`, raw-target, or old-profile tuning
  caches; rerun models/report for current-code log-target results.
- Completion-gap audit: `completion_gap_audit.json` has `ok=false`,
  `paper_grade_ready=false`, with 8 complete rows, 2 diagnostic-only rows, and
  2 incomplete rows.

Historical model snapshot key results, not current target-window claims:

- `jump_c2o`: the populated snapshot's old LightGBM/XGBoost ensemble row has
  best OOS R2 versus IVAR at 0.2374. Rerun models before citing the new
  dual-output forecast/rank ensemble row.
- `jump_c2o`: Goyal-Saretto spread has best MAE at 0.0075 and best AUC at
  0.6200.
- `day_c2c`: ridge-flat sequence has best OOS R2 versus IVAR at 0.2782, but
  this remains sequence-diagnostic evidence.
- `day_c2c`: Goyal-Saretto spread has best AUC at 0.6185 and the least-negative
  headline net proxy PnL at -1,948 USD.
- `reaction_o2c`: the populated snapshot's old LightGBM/XGBoost ensemble row
  has best OOS R2 versus IVAR at 0.9441, and ridge-flat sequence has best AUC
  at 0.8075.
- O2C strategy rows are diagnostic only and `pnl_headline_eligible=false`.

Broader preflight interpretation artifacts:

- `ivar_defeat_events.csv`: 4,260 rows.
- `ivar_defeat_metrics.csv`: 42 rows.
- `ivar_defeat_breakdowns.csv`: 3,804 rows.
- `casebook_events.csv`: 3,745 rows.
- `casebook_summary.csv`: 199 rows.
- `quote_confidence_prediction_coverage.csv`: 30 rows.
- `quote_confidence_strategy_summary.csv`: 72 rows.
- `quote_confidence_ivar_defeat_summary.csv`: 126 rows.
- `quote_confidence_casebook_summary.csv`: 531 rows.

## Current Sell Angle

The defensible near-term claim is a conservative signal-screening result:
the research design can measure earnings event-variance mispricing, compare
models against market IVAR and classical RV-IV spread, and identify where
tuned models improve variance-level fit. The current fast refresh does not
support positive headline C2C economics or executable trading performance.

Do not claim paper-grade executable performance, full-spread tradability, NBBO
evidence, feature-schema improvement versus retired profiles, sequence
superiority, or that lower RMSE alone proves economic value.

## Current Implementation Status

- PR #1 was squash-merged as `6ab8cb9`.
- The two PR Chinese contributor docs were deleted after their useful ideas
  were absorbed into code and paper-facing docs.
- `paper_plan.md` is the paper-style manuscript plan.
- `results_snapshot.md` is the paper-style Results and Discussion ledger.
- Quote-aware execution confidence, IVAR defeat analysis, and casebook
  artifacts are implemented.
- `just data` was rebuilt through contract-reference validation,
  trade-proxy panel, and lake-quality audit for the broader 2016-01-01
  preflight window. Full main-window quote/NBBO-equivalent coverage remains
  missing.
- `research --stage features` rebuilt the broader preflight gold feature
  matrix. Data/features/model/report outputs still need a 2016-10-01
  main-window rerun; the local Mac LightGBM runtime segfaulted at
  `lightgbm_tuned` during a no-sequence smoke run.

## Command Surface

Use `just` as the public command surface.

Key commands:

- `just status`
- `just check`
- `just data args="--dry-run"`
- `just data`
- `just research-fast`
- `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1_day_c2c_rank_log_rvar --feature-schema-version fe_v2_sec_xbrl"`
- `just docs`

## Credential and Portability Policy

Secrets are file paths outside the repo, for example `MASSIVE_API_KEY_FILE` and
`MASSIVE_FLAT_FILE_KEY_FILE`. Do not print, inline, or commit key values. Data
and uv environment locations should be device-specific `.env` settings, not
hard-coded repo paths.
