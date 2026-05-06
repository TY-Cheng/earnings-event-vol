# Earnings Event Vol

<!-- --8<-- [start:docs-home] -->
Reproducible research pipeline for U.S. equity-options earnings event variance
forecasting and risk-defined option backtests.

## Research Question

This is not a generic implied-volatility forecasting project. The paper-facing
question is:

> Can models improve trading decisions around option-implied earnings event
> variance mispricing?

The target is event-level realized earnings variance:

```text
RVAR_event = log(S_after / S_before)^2
```

The market benchmark is the event variance implied by short-dated options:

```text
IVAR_event
```

Ex post mispricing is:

```text
RVAR_event - IVAR_event
```

Trading decisions are evaluated in premium space. A raw variance forecast is
not enough; expected strategy value must beat market entry cost and transaction
cost estimates.

## Current State

Verified local state on 2026-05-06:

- `just data` builds the active no-NBBO proxy data pipeline.
- `just research` builds the proxy feature/model/report package from the
  current trade-proxy event panel.
- Current data range is `2022-12-01` through `2025-12-31`, because the observed
  Massive options day-aggregate entitlement in this workspace starts in 2022.
- The target paper range remains 2013-2025, but that needs upgraded historical
  option data entitlement or another licensed options route.
- All current trade-price results are `panel_grade=no_nbbo_trade_proxy` and
  `paper_grade=false`.

Latest proxy data artifacts:

- Dynamic calendar: 1,054 SEC-first candidate rows; 810 BMO/AMC main-sample
  candidates after universe and text-validation filters.
- Trade-proxy panel: 810 events, 801 with `RVAR_event`, 690 with trade-proxy
  `IVAR_event`.
- Proxy contracts: 12,038 candidates; 10,165 with usable pre-cutoff
  second-aggregate prices.
- Proxy straddle diagnostics: 779 rows; mean gross proxy PnL about 12.41 USD,
  mean haircut PnL about -159.98 USD.

Latest proxy modeling artifacts:

- Feature matrix: 810 rows.
- Models evaluated: market-implied IVAR, last-four RVAR, last-four IVAR,
  Goyal-Saretto-style RV-IV spread, Elastic Net, LightGBM, XGBoost,
  FT-Transformer, Mamba sequence encoder, and a mask-only Mamba ablation.
- Sequence audit: 678 eligible events out of 810 under the default path
  coverage rule; flagged as high sequence-selection risk.
- In the current no-NBBO proxy run, LightGBM and XGBoost look strongest on
  ranking and proxy strategy metrics. This is signal-screening evidence, not a
  paper-grade executable trading result.
- Proxy-Mamba is implemented but not a headline result in the current run.

## Command Surface

Use `just` as the public command surface:

```bash
just status
just check
just data args="--dry-run"
just data
just research args="--allow-high-sequence-risk"
just docs
```

`just check` formats, fixes lint, runs mypy, pytest, MkDocs strict build,
status, and source probes.

`just data` runs:

```text
options-day-aggs-bulk -> universe -> dynamic-calendar -> pilot-panel -> trade-proxy-panel
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
  - underlying stock day aggregates for underlying closes, event returns,
    `RVAR_event`, and exit spot;
  - targeted Massive option second aggregates from
    `/range/1/second/<date>/<date>` for the entry proxy.
- entry proxy window: keep only bars in the resolved pre-cutoff buffer,
  default 60 minutes before the event cutoff, then select the latest positive
  `option_vwap` or `option_close` in the final 900 seconds.
- second aggregates are trade OHLCV bars, not quote, bid/ask, or NBBO data;
  exit proxy uses same-contract option day-aggregate close when available.

`just research` does not download market data. It consumes the current proxy
panel, builds features, trains/evaluates models, and writes metrics, figures,
and the proxy report.

## Key Outputs

Data pipeline:

- `artifacts/data_pipeline/data_pipeline_manifest.json`
- `artifacts/data_pipeline/universe/universe_manifest.json`
- `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json`
- `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json`
- `data/gold/event_panel/trade_proxy_event_panel.parquet`

Research package:

- `data/gold/modeling/feature_matrix.parquet`
- `artifacts/modeling/forecast_metrics.csv`
- `artifacts/modeling/ranking_metrics.csv`
- `artifacts/modeling/strategy_metrics.csv`
- `artifacts/modeling/model_fit_diagnostics.csv`
- `artifacts/modeling/model_predictions.parquet`
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
> the market-implied IVAR baseline. Paper-grade claims require quote/NBBO data
> and robust cost/inference checks.

## Docs

- Home: project object and current status.
- Results Snapshot: current artifacts and readiness boundaries.
- Paper Plan: research design and model/backtest protocol.
- Audit Prompts: implementation and manuscript review checklists.
- Future Work: paper blockers and deferred extensions.

`SPEC.md` is the implementation and research-protocol contract. It stays at the
repo root and is not a separate docs-nav page.
<!-- --8<-- [end:docs-home] -->
