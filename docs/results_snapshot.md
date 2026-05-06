# Results Snapshot

This page is the curated reader-facing snapshot. Raw generated outputs live
under ignored `artifacts/`, `data/`, and `reports/` paths; only selected
results and figures are copied into `docs/` for publication.

## Status

Current local run state, verified on 2026-05-06:

| Item | Current state |
| --- | --- |
| Data route | Massive option second aggregates plus option day aggregates |
| Execution grade | `no_nbbo_trade_proxy`, `paper_grade=false` |
| Study window in current run | 2022-12-01 to 2025-12-31 |
| Target paper window | 2013-2025, pending historical quote/NBBO or equivalent data |
| Universe | Monthly top 50 liquid U.S. single-name option underlyings |
| Event source | SEC EDGAR submissions plus SEC primary filing document text |
| Main timing sample | BMO and AMC only |
| Research package | Feature matrix, model metrics, proxy strategy diagnostics, figures |

The current evidence is useful for engineering validation and signal screening.
It is not paper-grade execution evidence because it does not use historical
bid/ask or NBBO quotes.

## Data Coverage

Latest proxy data pipeline outputs:

| Measure | Value |
| --- | ---: |
| Dynamic-calendar rows | 1,054 |
| BMO/AMC main-sample candidates | 810 |
| Trade-proxy event-panel rows | 810 |
| Events with `RVAR_event` | 801 |
| Events with trade-proxy `IVAR_event` | 690 |
| Proxy contract candidates | 12,038 |
| Contracts with usable pre-cutoff proxy price | 10,165 |
| Contracts with no trade in cutoff window | 1,873 |
| Main DTE 5-14 contracts | 5,098 |
| Robustness DTE 3-21 contracts | 12,038 |
| Proxy straddle diagnostic rows | 779 |

IVAR failure diagnostics:

| Failure reason | Events |
| --- | ---: |
| No two event-covering expiries | 103 |
| Nonmonotone total variance | 10 |
| Negative extracted IVAR | 7 |

Proxy straddle diagnostics:

| Measure | Value |
| --- | ---: |
| Mean gross proxy PnL | 12.41 USD |
| Mean haircut proxy PnL | -159.98 USD |

Interpretation: the proxy route produces a usable event panel, but the IVAR
coverage gap is still material. The main loss channel is missing a valid pair of
event-covering expiries, followed by smaller term-structure extraction failures.

## Model Results

The feature matrix has 810 rows. The current chronological proxy split trains on
567 rows, validates on 121 rows, and tests on 122 rows for tabular models. Mamba
uses the sequence-eligible subset: 475 train rows, 103 validation rows, and 100
test rows.

Selected forecast metrics:

| Model | N | MAE | RMSE | OOS R2 vs IVAR |
| --- | ---: | ---: | ---: | ---: |
| Market IVAR | 99 | 0.0106 | 0.0167 | 0.000 |
| Goyal-Saretto spread | 99 | 0.0105 | 0.0173 | -0.077 |
| Elastic Net | 122 | 0.0113 | 0.0258 | 0.401 |
| LightGBM | 122 | 0.0075 | 0.0194 | 0.677 |
| XGBoost | 122 | 0.0066 | 0.0198 | 0.525 |
| FT-Transformer | 122 | 0.0365 | 0.0406 | -3.857 |
| Mamba sequence encoder | 100 | 0.0094 | 0.0174 | -0.005 |

Selected ranking metrics:

| Model | N | Top-decile precision | AUC | Brier |
| --- | ---: | ---: | ---: | ---: |
| Market IVAR | 99 | 0.000 | 0.500 | 0.252 |
| Last-four RVAR | 99 | 0.400 | 0.551 | 0.316 |
| Goyal-Saretto spread | 99 | 0.400 | 0.571 | 0.308 |
| Elastic Net | 99 | 0.900 | 0.822 | 0.211 |
| LightGBM | 99 | 1.000 | 0.975 | 0.152 |
| XGBoost | 99 | 0.900 | 0.959 | 0.158 |
| FT-Transformer | 99 | 0.200 | 0.464 | 0.350 |
| Mamba sequence encoder | 86 | 0.111 | 0.458 | 0.353 |

Selected proxy strategy metrics:

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Last-four RVAR | 99 | 4,801.84 | 0.028 | 0.379 | -11,227.73 |
| Last-four IVAR | 99 | -2,066.32 | -0.012 | -0.163 | -8,924.75 |
| Goyal-Saretto spread | 99 | 3,169.84 | 0.019 | 0.250 | -20,002.03 |
| Elastic Net | 99 | 51,678.28 | 0.306 | 4.511 | -1,330.81 |
| LightGBM | 99 | 69,694.96 | 0.413 | 6.690 | -442.37 |
| XGBoost | 99 | 64,352.96 | 0.381 | 5.975 | -1,191.77 |
| FT-Transformer | 99 | 2,618.64 | 0.016 | 0.207 | -10,414.40 |
| Mamba sequence encoder | 86 | -3,651.72 | -0.026 | -0.301 | -11,336.97 |

Interpretation: in this no-NBBO proxy run, the strongest evidence is ranking,
not generic variance RMSE. LightGBM and XGBoost dominate the market IVAR
baseline and the simple historical baselines on top-decile precision and AUC.
Mamba is implemented and has a sequence route, but it is not the headline model
in the current run.

The market-implied IVAR baseline is still the central benchmark. It generates no
trades under the premium-edge rule because its forecast edge is zero by
construction.

## Figures

Forecast error:

![Forecast performance](assets/images/modeling/forecast_performance.png)

Ranking quality:

![AUC and top-decile precision](assets/images/modeling/auc_top_decile_precision.png)

Proxy strategy PnL:

![Strategy PnL by edge decile](assets/images/modeling/strategy_pnl_by_edge_decile.png)

Cost sensitivity:

![Cost sensitivity](assets/images/modeling/cost_sensitivity.png)

Calibration:

![Calibration plot](assets/images/modeling/calibration_plot.png)

Realized mispricing by predicted edge decile:

![Edge decile realized mispricing](assets/images/modeling/edge_decile_realized_mispricing.png)

## What This Means

Current proxy-stage takeaways:

- The pipeline is now beyond toy smoke tests: it has a dynamic top-50 universe,
  SEC-first event validation, an event panel, a feature matrix, model metrics,
  proxy strategy diagnostics, and figures.
- The signal-screening result is encouraging for tabular nonlinear models,
  especially LightGBM and XGBoost.
- The Mamba route is present but currently weaker than the tabular baselines.
  It needs sequence-selection diagnostics and robustness before it can be a
  paper claim.
- No current result supports full-spread executable trading claims.

The defensible near-term claim is:

> In a no-NBBO proxy sample, state and event-history features show preliminary
> cross-sectional ranking signal for earnings event-variance mispricing beyond
> the market-implied IVAR baseline.

The paper-grade claim requires:

- historical bid/ask or NBBO-equivalent option data;
- quote-based IVAR and leg-level strategy construction;
- full bid-ask crossing as the main cost assumption;
- robustness across DTE windows, years, liquidity regimes, and BMO/AMC timing;
- clustered or bootstrap inference.

## Artifact Map

Local raw outputs:

| Purpose | Path |
| --- | --- |
| Data pipeline manifest | `artifacts/data_pipeline/data_pipeline_manifest.json` |
| Universe manifest | `artifacts/data_pipeline/universe/universe_manifest.json` |
| Dynamic calendar report | `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json` |
| Trade-proxy panel report | `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json` |
| Feature matrix | `data/gold/modeling/feature_matrix.parquet` |
| Forecast metrics | `artifacts/modeling/forecast_metrics.csv` |
| Ranking metrics | `artifacts/modeling/ranking_metrics.csv` |
| Strategy metrics | `artifacts/modeling/strategy_metrics.csv` |
| Model diagnostics | `artifacts/modeling/model_fit_diagnostics.csv` |
| Proxy report | `reports/modeling/proxy_research_report.md` |

Published docs assets:

| Purpose | Path |
| --- | --- |
| Curated results page | `docs/results_snapshot.md` |
| Published figure copies | `docs/assets/images/modeling/*.png` |

## Docs Structure

The reader-facing docs intentionally stay small:

- Home: short project overview and current status.
- Results Snapshot: current curated data/model/proxy-strategy results and
  analysis.
- Paper Plan: research protocol, target variables, model ladder, and evaluation
  design.
- Audit Prompts: implementation and manuscript review checklists.
- Future Work: paper blockers and deferred extensions.

`SPEC.md` remains the repo-root implementation contract and is not a separate
nav page.
