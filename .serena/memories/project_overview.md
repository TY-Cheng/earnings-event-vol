# Project Overview

**Name**: earnings-event-vol

**Purpose**: Reproducible research pipeline for U.S. equity-options earnings
event variance forecasting and risk-defined option backtests.

**Core research question**: Can models improve trading decisions around
option-implied earnings event variance mispricing? This is not generic
implied-volatility forecasting.

## Research Contract

- Treat `SPEC.md` as the implementation and research-protocol contract.
- Models forecast realized earnings event variance labels.
- Market baseline: `IVAR_event` extracted from short-dated options.
- C2C ex post mispricing: `RVAR_event_day_c2c - IVAR_event`.
- Trade entry uses premium-space expected edge, not raw variance edge.
- Current results are `no_nbbo_trade_proxy` and `paper_grade=false`.

## Target System

- `jump_c2o`: primary scientific target, close-to-open earnings jump variance.
- `day_c2c`: literature-compatible target and the only V1 proxy-PnL headline.
- `reaction_o2c`: post-open digestion diagnostic.

## Current Data State

- Current proxy window: 2022-12-01 through 2025-12-31.
- Target paper window: 2013-2025, pending historical quote/NBBO or equivalent
  data.
- Dynamic monthly top-50 liquid U.S. single-name option underlyings.
- SEC EDGAR 8-K Item 2.02 discovery with SEC primary-document text validation.
- SEC CompanyFacts XBRL fundamentals with conservative as-of gating.
- Massive options day aggregates, underlying day aggregates, and targeted
  option one-second trade aggregates.
- Latest proxy panel: 810 BMO/AMC events; 801 with C2C `rvar_event`; 693 with
  trade-proxy `IVAR_event`; 12,038 proxy contract candidates; 10,165 usable
  pre-cutoff prices.

## Feature Protocol

- Default schema: `fe_v2_sec_xbrl`.
- Ablation schema: `fe_v1_legacy`.
- Resolved allowlist: `artifacts/modeling/feature_schema_report.csv`.
- Transform artifact: `artifacts/modeling/feature_transform_params.json`.
- Active FE V2 adds point-in-time rolling earnings history, SEC XBRL
  fundamentals, train-fitted rank/z-score transforms, and single-name
  run-up/surface proxy features.
- If FE V2 underperforms FE V1 on locked test, report it as a diagnostic
  negative result instead of cherry-picking.

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
- LightGBM/XGBoost rank-average ensemble
- FT-Transformer

Sequence diagnostics:

- ridge-flat sequence aggregates
- BiGRU 5-seed ensemble
- official bidirectional `mamba-ssm` 5-seed ensemble
- attention pooling
- non-causal dilated CNN
- mask-only and deterministic time-shuffle controls

Retired legacy ids:

- `daily_mamba_20step`
- `hybrid_mamba_31step`
- `intraday_only_mamba_12step`
- `mask_only_hybrid_mamba`

These legacy ids were retired because they used an in-repo gated recurrent
encoder rather than official `mamba-ssm`.

## Current Result Summary

- Current canonical FE V2 run:
  `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v2_sec_xbrl"`.
- Active FE V2 is not the current sell. Goyal-Saretto spread has the strongest
  `jump_c2o` AUC at about 0.602; LightGBM has best `jump_c2o` OOS R2 versus
  IVAR at about 0.203.
- FE V2 ridge-flat sequence aggregates have positive `day_c2c` proxy PnL of
  about 19,918 USD, but the row is diagnostic because sequence selection risk
  remains high.
- Same-code FE V1 ablation is stronger: LightGBM has `jump_c2o` AUC about
  0.677 and leads `day_c2c` headline proxy PnL at about 53,664 USD; XGBoost has
  best `jump_c2o` OOS R2 versus IVAR at about 0.375.
- `reaction_o2c` remains diagnostic-only. Ridge-flat sequence leads O2C AUC at
  about 0.799 in both schemas, but full-event `IVAR_event` is a weak O2C
  comparator and all O2C strategy rows are `pnl_headline_eligible=false`.
- The full sequence suite did not pass the common-row bootstrap gate. Official
  `mamba-ssm` 5-seed has `jump_c2o` AUC about 0.501 and negative `day_c2c`
  proxy PnL in the FE V2 active run.

## Current Sell Angle

The defensible near-term claim is: in a no-NBBO proxy sample, a parsimonious
event-level tabular feature set shows preliminary cross-sectional ranking
signal for earnings event-variance mispricing beyond the market-implied IVAR
baseline, and the best FE V1 tabular model maps that ranking signal into
positive premium-space proxy economics. FE V2 is currently a negative
diagnostic result.

Do not claim paper-grade executable performance, Mamba superiority,
full-spread tradability, NBBO evidence, FE V2 improvement, or that lower RMSE
alone proves economic value.

## Command Surface

Use `just` as the public command surface. Keep it small and prefer
parameterized `just data ...` variants over new top-level recipes.

Key commands:

- `just status`
- `just check`
- `just mamba-doctor`
- `just mamba-install`
- `just data args="--dry-run"`
- `just data`
- `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v2_sec_xbrl"`
- `just docs`

## Credential and Portability Policy

Secrets are file paths outside the repo, for example `MASSIVE_API_KEY_FILE`
and `MASSIVE_FLAT_FILE_KEY_FILE`. Do not print, inline, or commit key values.
Data and uv environment locations should be device-specific `.env` settings,
not hard-coded repo paths.
