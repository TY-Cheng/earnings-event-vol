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

Verified local state on 2026-05-07:

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
- Trade-proxy panel: 810 events, 801 with the backward-compatible C2C
  `rvar_event` alias, 693 with trade-proxy `IVAR_event`.
- Proxy contracts: 12,038 candidates; 10,165 with usable pre-cutoff
  second-aggregate prices.
- Proxy straddle diagnostics: 779 rows; mean gross C2C primary exit-preclose
  VWAP proxy PnL about -100.72 USD, mean haircut proxy PnL about -250.54 USD.

Latest proxy modeling artifacts:

- Feature matrix: 810 rows.
- Models evaluated: market-implied IVAR, last-four RVAR, last-four IVAR,
  Goyal-Saretto-style RV-IV spread, Elastic Net, LightGBM, XGBoost,
  FT-Transformer, daily proxy-Mamba, hybrid proxy-Mamba, intraday-only Mamba,
  and mask-only Mamba ablations.
- Sequence audit: 678 eligible events out of 810 under the default path
  coverage rule; flagged as high sequence-selection risk.
- In the current no-NBBO proxy run, XGBoost leads `jump_c2o` ranking AUC
  (0.781), while LightGBM leads `day_c2c` net proxy PnL (about 69,908 USD).
  This is signal-screening evidence, not a paper-grade executable trading
  result.
- Proxy-Mamba is implemented for both the 20-step daily tensor and the 31-step
  hybrid tensor, but it is not a headline model in the current run.

## Command Surface

Use `just` as the public command surface:

```bash
just status
just check
just data args="--dry-run"
just data
just research args="--allow-high-sequence-risk --split-design chronological_proxy_70_15_15"
just docs
```

`just check` formats, fixes lint, runs mypy, pytest, MkDocs strict build,
status, and source probes.

`just data` runs the active proxy-all DAG:

```text
options-day-aggs-bulk -> universe -> dynamic-calendar -> event-window-panel
  -> contract-reference-validation -> trade-proxy-panel
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
> the market-implied IVAR baseline, and the best tabular models map that ranking
> signal into positive premium-space proxy economics. Paper-grade claims require
> quote/NBBO data and robust cost/inference checks.

## Docs

- Home: project object and current status.
- Results Snapshot: current artifacts and readiness boundaries.
- Paper Plan: research design and model/backtest protocol.
- Audit Prompts: implementation and manuscript review checklists.
- Future Work: paper blockers and deferred extensions.

`SPEC.md` is the implementation and research-protocol contract. It stays at the
repo root and is not a separate docs-nav page.
<!-- --8<-- [end:docs-home] -->
