# Earnings Event Vol

<!-- --8<-- [start:docs-home] -->
Reproducible research pipeline for U.S. equity-options earnings event variance
forecasting and risk-defined option backtests.

## Research Question

This is not a generic implied-volatility forecasting project. The paper-facing
question is:

> Can models improve trading decisions around option-implied earnings event
> variance mispricing?

The realized-variance target system is decomposed into three labels:

```text
jump_c2o     = close-to-open earnings jump variance
day_c2c      = close-to-close full reaction-day variance
reaction_o2c = open-to-close post-open digestion variance
```

The market benchmark is the event variance implied by short-dated options:

```text
IVAR_event
```

C2C ex post mispricing is:

```text
RVAR_event_day_c2c - IVAR_event
```

The V1 strategy/PnL layer uses `day_c2c` only. `jump_c2o` is the primary
scientific forecast/ranking target, but it is not reported as executable option
PnL in the current no-NBBO proxy run. Trading decisions are evaluated in premium
space. A raw variance forecast is not enough; expected strategy value must beat
market entry cost and transaction cost estimates.

## Current State

Verified local state on 2026-05-12:

- `just data` builds the active no-NBBO proxy data pipeline.
- `just research` builds the canonical V5 proxy feature/model/report
  package from the current trade-proxy event panel. The current paper-facing
  snapshot uses the canonical tuned protocol.
- `just mamba-install` installs the local CUDA Mamba wheels and `just
  mamba-doctor` verifies the official `mamba-ssm` runtime.
- Current data range is `2022-12-01` through `2025-12-31`, because the observed
  Massive options day-aggregate entitlement in this workspace starts in 2022.
- The target paper range remains 2013-2025, but that needs upgraded historical
  option data entitlement or another licensed options route.
- All current trade-price results are `panel_grade=no_nbbo_trade_proxy` and
  `paper_grade=false`.

Latest proxy data artifacts:

- Dynamic calendar: 1,054 SEC-first candidate rows; 810 BMO/AMC main-sample
  candidates after universe and text-validation filters.
- Trade-proxy panel: 810 events, 801 with the backward-compatible C2C
  `rvar_event` alias, 693 with trade-proxy `IVAR_event`.
- Proxy contracts: 12,038 candidates; 10,165 with usable pre-cutoff
  second-aggregate prices.
- Proxy straddle diagnostics: 779 rows; mean gross C2C primary exit-preclose
  VWAP proxy PnL about -100.72 USD, mean haircut proxy PnL about -250.54 USD.

Latest proxy modeling artifacts:

- Feature matrix: 810 rows.
- Models evaluated: market-implied IVAR, last-four RVAR, last-four IVAR,
  Goyal-Saretto-style RV-IV spread, Elastic Net, LightGBM, XGBoost, a
  LightGBM/XGBoost rank-average ensemble, FT-Transformer, and the V5 sequence
  diagnostic suite.
- Current tuned protocol: the canonical tuned-only research protocol.
  Hyperparameter selection uses train and locked-validation rows only, then
  evaluates locked test rows once. Paired original tabular and single-seed
  sequence rows are no longer emitted.
- Full sequence diagnostic suite: ridge-flat sequence aggregates, 5-seed BiGRU,
  5-seed official bidirectional `mamba-ssm`, attention pooling, non-causal
  dilated CNN, mask-only, and deterministic time-shuffle controls.
- Sequence audit: 678 eligible events out of 810 under the default path
  coverage rule; flagged as high sequence-selection risk.
- `FT-Transformer` refers to the validation-tuned tabular transformer
  specification.
- The active canonical outputs use the default `fe_v2_sec_xbrl` schema, but
  the same-code FE V1 versus FE V2 ablation is negative for FE V2. In FE V2,
  the strongest `jump_c2o` AUC is the Goyal-Saretto-style spread at about
  0.602, and the positive `day_c2c` ridge-flat sequence PnL of about 19,918
  USD is diagnostic because the sequence gate does not pass.
- The stronger current sell is the `fe_v1_legacy` same-code ablation:
  LightGBM reaches `jump_c2o` AUC about 0.677, XGBoost has best `jump_c2o`
  OOS R2 versus IVAR at about 0.375, and LightGBM leads the `day_c2c` headline
  proxy strategy at about 53,664 USD net PnL. This is signal-screening
  evidence, not a paper-grade executable trading result.
- `reaction_o2c` is now included in the V5 proxy model artifacts as a
  diagnostic target. Ridge-flat sequence aggregates lead O2C AUC at about
  0.799; among the tabular rows, XGBoost leads at about 0.768. O2C uses
  full-event `IVAR_event` only as a weak comparator and all O2C strategy rows
  remain `pnl_headline_eligible=false`.
- The full sequence diagnostic suite has not passed the common-row bootstrap
  gate in the current proxy evidence: the 5-seed official `mamba-ssm` row has
  `jump_c2o` AUC about 0.501 and negative `day_c2c` proxy PnL. Sequence rows
  remain diagnostic and do not upgrade the claim.

## Command Surface

Use `just` as the public command surface:

```bash
just status
just check
just mamba-doctor
just mamba-install
just data args="--dry-run"
just data
just research
just research-report
just docs
```

`just check` formats, fixes lint, runs mypy, pytest, MkDocs strict build,
status, and source probes.

`just data` runs the active proxy-all DAG:

```text
options-day-aggs-bulk -> universe -> dynamic-calendar -> sec-companyfacts
  -> event-window-panel -> contract-reference-validation -> trade-proxy-panel
```

Default data parameters:

- study range: `2022-12-01` to `2025-12-31`;
- universe lookback: from `2022-06-01`;
- monthly top 50 liquid U.S. single-name option underlyings;
- DTE `3-21`, supporting the main `5-14` sample and robustness window;
- market data route:
  - options day aggregates for universe liquidity ranking, contract discovery,
    local IV/IVAR proxy inputs, same-contract option exit closes, and the
    20-day close-trade-implied option-surface sequence;
  - underlying stock day aggregates for underlying closes, vendor OHLC opens,
    C2O/C2C/O2C event returns, and exit spot;
  - targeted Massive option second aggregates from
    `/range/1/second/<date>/<date>` for the entry proxy.
- entry proxy window: keep only bars in the resolved pre-cutoff buffer,
  default 60 minutes before the event cutoff, then compute the true per-leg
  volume-weighted `option_vwap` over the final 900 seconds.
- The option-proxy open anchor is unified as same-contract option VWAP from
  5-15 minutes after open. C2O uses it as the primary post-open exit proxy;
  O2C uses the same mark as the diagnostic post-open entry proxy. The 0-5
  minute VWAP remains an opening-microstructure stress test.
- second aggregates are trade OHLCV bars, not quote, bid/ask, or NBBO data;
  the primary C2C exit proxy is same-contract option VWAP over the final
  15 minutes before the exit-date close. Same-contract option day-aggregate
  close is retained only as fallback/diagnostic.
- SEC CompanyFacts is public XBRL financial-statement data. The active stage
  uses CIK-mapped CompanyFacts with conservative as-of gating:
  `acceptanceDateTime <= feature_asof_timestamp` when available, otherwise
  `filed < feature_asof_date`.

`just research` does not download market data. For the paper-facing snapshot,
run the canonical tuned proxy package with the full sequence diagnostic suite
and 1,000 bootstrap iterations:

```bash
just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v2_sec_xbrl"
```

In the canonical tuned protocol, Optuna objectives and `ElasticNetCV` read only train and
locked validation rows. The selected hyperparameters are refit on
train+validation, and locked test rows are evaluated once after selection.
Paired original rows are intentionally not emitted.

The default feature schema is `fe_v2_sec_xbrl`. It uses the resolved
`artifacts/modeling/feature_schema_report.csv` as the model-feature allowlist,
excludes raw IDs and outcome/exit/PnL fields, adds point-in-time rolling
same-ticker earnings history, SEC XBRL fundamentals, train-fitted rank/z-score
features, and single-name run-up/surface proxy features. `fe_v1_legacy` remains
available only for same-code feature-ablation reruns.

It consumes the current proxy panel, builds features, trains/evaluates models,
writes metrics, writes `reports/modeling/proxy_research_report.md`, regenerates
`reports/modeling/figures/*.png`, and syncs those figures into
`docs/assets/images/modeling/`.

`just research-report` regenerates only the generated report and figure assets
from existing modeling artifacts. The curated reader-facing
`docs/results_snapshot.md` is intentionally manual: update it when a run changes
the paper-facing tables or interpretation, then run `just check`.

## Key Outputs

Data pipeline:

- `artifacts/data_pipeline/data_pipeline_manifest.json`
- `artifacts/data_pipeline/universe/universe_manifest.json`
- `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json`
- `artifacts/data_pipeline/sec_companyfacts/sec_companyfacts_manifest.json`
- `artifacts/data_pipeline/sec_companyfacts/sec_companyfacts_diagnostics.csv`
- `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json`
- `$GOLD_DATA_DIR/event_panel/trade_proxy_event_panel.parquet`

Research package:

- `$GOLD_DATA_DIR/modeling/feature_matrix.parquet`
- `artifacts/modeling/feature_schema_report.csv`
- `artifacts/modeling/feature_transform_params.json`
- `artifacts/modeling/forecast_metrics.csv`
- `artifacts/modeling/ranking_metrics.csv`
- `artifacts/modeling/strategy_metrics.csv`
- `artifacts/modeling/model_fit_diagnostics.csv`
- `artifacts/modeling/model_predictions.parquet`
- `artifacts/modeling/sequence_v2_quality.csv`
- `artifacts/modeling/common_row_pairwise_metrics.csv`
- `artifacts/modeling/incremental_value_diagnostics.csv`
- `artifacts/modeling/sequence_model_fit_diagnostics.csv`
- `reports/modeling/proxy_research_report.md`
- `reports/modeling/figures/`

## Claim Boundaries

Current evidence supports engineering and signal-screening discussion only.
It does not support final paper claims that require bid/ask or NBBO execution.

Do not claim:

- generic IV forecasting superiority;
- paper-grade full-spread tradability;
- that second-aggregate trade bars are NBBO quotes;
- that Mamba is the contribution independent of baselines and costs;
- that lower RMSE alone implies economic value.

The defensible near-term claim is narrower:

> In a no-NBBO proxy sample, state and event-history features show preliminary
> cross-sectional ranking signal for earnings event-variance mispricing beyond
> the market-implied IVAR baseline. The current same-code ablation says the
> parsimonious FE V1 tabular signal is stronger than the richer FE V2 default,
> so FE V2 is a negative diagnostic result rather than a headline improvement.
> Paper-grade claims require quote/NBBO data and robust cost/inference checks.

## Docs

- Home: project object and current status.
- Results Snapshot: current artifacts and readiness boundaries.
- Paper Plan: research design and model/backtest protocol.
- Audit Prompts: implementation and manuscript review checklists.
- Future Work: paper blockers and deferred extensions.

`SPEC.md` is the implementation and research-protocol contract. It stays at the
repo root and is not a separate docs-nav page.
<!-- --8<-- [end:docs-home] -->
