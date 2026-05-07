---
hide:
  - navigation
---

# Results Snapshot

This page is the curated reader-facing snapshot. Raw generated outputs live
under ignored `artifacts/`, `data/`, and `reports/` paths; selected figures are
copied into `docs/` for publication.

## Status

Current local run state, verified on 2026-05-07 from
`artifacts/modeling/*`, `artifacts/data_pipeline/*`, and
`reports/modeling/proxy_research_report.md`:

| Item | Current state |
| --- | --- |
| Data route | SEC-first event calendar plus Massive market-data proxy |
| Execution grade | `no_nbbo_trade_proxy`, `paper_grade=false` |
| Current study window | 2022-12-01 to 2025-12-31 |
| Target paper window | 2013-2025, pending historical quote/NBBO or equivalent data |
| Universe | Monthly top 50 liquid U.S. single-name option underlyings |
| Event source | SEC EDGAR submissions plus SEC primary filing document text |
| Main timing sample | BMO and AMC only |
| Research package | C2O/C2C/O2C feature matrix, model metrics, proxy strategy diagnostics, figures |

The current evidence is suitable for engineering validation, signal screening,
and a conservative draft of the empirical story. It is not paper-grade execution
evidence because it does not use historical bid/ask or NBBO quotes.

## Data and Backtest Setup

The proxy route uses:

- **Options day aggregates** for dynamic-universe liquidity ranking, contract
  discovery, local IV/IVAR proxy inputs, fallback diagnostics, and the daily
  close-trade-implied option-surface sequence.
- **Underlying stock day aggregates** for underlying closes, vendor OHLC opens,
  C2O/C2C/O2C event returns, and exit spot.
- **Option one-second trade aggregates** for targeted pre-cutoff entry prices,
  primary C2C exit prices, and post-open C2O/O2C anchor prices. These are trade
  OHLCV bars, not quote midpoints, bid/ask records, or NBBO.

Entry pricing uses per-leg option VWAP over the final 900 seconds before the
event cutoff. The primary C2C exit uses same-contract option VWAP over the final
15 minutes before the exit-date close. The C2O diagnostic exit and O2C diagnostic
entry use same-contract option VWAP from 5-15 minutes after the regular-session
open; 0-5 minutes is retained as an opening-microstructure stress test.

The target system is:

- `jump_c2o`: primary scientific target, close-to-open earnings jump variance.
- `day_c2c`: literature-compatible target and the only V1 proxy-PnL headline.
- `reaction_o2c`: post-open digestion diagnostic.

The market baseline is `IVAR_event`. C2C ex post mispricing is
`RVAR_event_day_c2c - IVAR_event`. Strategy entry is evaluated in premium space,
not from raw variance edge alone.

## Data Coverage

Latest proxy data pipeline outputs:

| Measure | Value |
| --- | ---: |
| Dynamic-calendar rows | 1,054 |
| BMO/AMC main-sample candidates | 810 |
| Trade-proxy event-panel rows | 810 |
| Events with C2C `rvar_event` alias | 801 |
| Events with trade-proxy `IVAR_event` | 693 |
| Proxy contract candidates | 12,038 |
| Contracts with usable pre-cutoff proxy price | 10,165 |
| Contracts with no trade in cutoff window | 1,873 |
| Contracts with local IV proxy | 10,138 |
| Main DTE 5-14 contracts | 5,098 |
| Robustness DTE 3-21 contracts | 12,038 |
| Proxy straddle diagnostic rows | 779 |

IVAR failure diagnostics:

| Failure reason | Events |
| --- | ---: |
| No two event-covering expiries | 103 |
| Nonmonotone total variance | 7 |
| Negative extracted IVAR | 7 |

Proxy straddle diagnostics:

| Measure | Value |
| --- | ---: |
| Mean gross proxy PnL, all recorded marks | -78.97 USD |
| Mean gross C2C primary exit-preclose VWAP proxy PnL | -100.72 USD |
| Mean haircut proxy PnL | -250.54 USD |
| Mean C2O intrinsic-open gross diagnostic PnL | -384.24 USD |
| Mean C2O post-open option VWAP 0-5m proxy PnL | -21.14 USD |
| Mean C2O post-open option VWAP 5-15m proxy PnL | -7.31 USD |
| Mean O2C option VWAP 5-15m to primary C2C exit diagnostic PnL | -24.31 USD |
| Mean option-proxy decomposition residual, 5-15m | 0.00 USD |

Interpretation: the event panel is large enough for proxy-stage model
comparison, but IVAR coverage remains a material screen. The main IVAR loss
channel is missing two event-covering expiries.

## Model Results

The feature matrix has 810 rows. The chronological proxy split trains on 567
rows, validates on 121 rows, and tests on 122 rows for tabular models. Daily
Mamba uses a 20 x 37 tensor; hybrid Mamba uses a 31 x 21 mixed-clock tensor with
19 prior daily proxy-surface states plus 12 entry-day five-minute trade-aggregate
proxy bins. Sequence coverage is 678 eligible events out of 810, so Mamba
results are diagnostic rather than headline evidence.

Selected `jump_c2o` forecast metrics:

| Model | N | MAE | RMSE | OOS R2 vs IVAR |
|:---|---:|---:|---:|---:|
| Market IVAR | 100 | 0.0097 | 0.0145 | 0.000 |
| Goyal-Saretto spread | 100 | 0.0076 | 0.0134 | 0.141 |
| Elastic Net | 122 | 0.0095 | 0.0213 | 0.323 |
| LightGBM | 122 | 0.0077 | 0.0192 | 0.355 |
| XGBoost | 122 | 0.0074 | 0.0191 | 0.380 |
| FT-Transformer | 122 | 0.0374 | 0.0396 | -5.487 |
| Daily Mamba 20-step | 100 | 0.0067 | 0.0136 | 0.106 |
| Hybrid Mamba 31-step | 100 | 0.0082 | 0.0228 | 0.194 |
| Intraday-only Mamba 12-step | 100 | 0.0083 | 0.0228 | 0.192 |
| Mask-only hybrid Mamba | 100 | 0.0088 | 0.0242 | -0.036 |

Selected `jump_c2o` ranking metrics:

| Model | N | Top-decile precision | AUC | Edge-decile Spearman |
|:---|---:|---:|---:|---:|
| Market IVAR | 100 | 0.000 | 0.500 | -0.006 |
| Goyal-Saretto spread | 100 | 0.300 | 0.602 | 0.685 |
| Elastic Net | 100 | 0.500 | 0.629 | 0.855 |
| LightGBM | 100 | 0.500 | 0.745 | 0.903 |
| XGBoost | 100 | 0.500 | 0.781 | 0.927 |
| FT-Transformer | 100 | 0.200 | 0.525 | 0.758 |
| Daily Mamba 20-step | 87 | 0.111 | 0.495 | 0.406 |
| Hybrid Mamba 31-step | 93 | 0.100 | 0.498 | 0.600 |
| Intraday-only Mamba 12-step | 93 | 0.100 | 0.498 | 0.673 |
| Mask-only hybrid Mamba | 93 | 0.100 | 0.500 | 0.685 |

Selected `day_c2c` proxy strategy metrics:

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|---:|---:|---:|---:|---:|
| Last-four RVAR | 100 | -3,481.69 | -0.021 | -0.268 | -16,788.67 |
| Last-four IVAR | 100 | -15,904.43 | -0.094 | -1.235 | -16,131.89 |
| Goyal-Saretto spread | 100 | -461.11 | -0.003 | -0.036 | -17,144.74 |
| Elastic Net | 100 | 47,937.75 | 0.283 | 4.007 | -3,201.65 |
| LightGBM | 100 | 69,908.11 | 0.413 | 6.476 | -1,416.38 |
| XGBoost | 100 | 68,343.93 | 0.403 | 6.271 | -1,694.94 |
| FT-Transformer | 100 | -4,792.98 | -0.028 | -0.370 | -15,366.08 |
| Daily Mamba 20-step | 87 | -9,370.00 | -0.066 | -0.765 | -14,011.80 |
| Hybrid Mamba 31-step | 93 | -34.55 | -0.000 | -0.003 | -11,928.45 |
| Intraday-only Mamba 12-step | 93 | -34.55 | -0.000 | -0.003 | -11,928.45 |
| Mask-only hybrid Mamba | 93 | 101.21 | 0.001 | 0.008 | -11,928.45 |

Selected `jump_c2o` 5-15 minute post-open option-VWAP proxy diagnostics:

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|---:|---:|---:|---:|---:|
| Last-four RVAR | 93 | -804.97 | -0.005 | -0.074 | -10,016.79 |
| Goyal-Saretto spread | 93 | -1,531.36 | -0.009 | -0.141 | -12,854.27 |
| Elastic Net | 93 | 14,324.21 | 0.085 | 1.335 | -5,844.42 |
| LightGBM | 93 | 28,911.28 | 0.171 | 2.782 | -3,909.83 |
| XGBoost | 93 | 41,455.71 | 0.245 | 4.190 | -1,697.70 |
| FT-Transformer | 93 | 4,112.63 | 0.024 | 0.380 | -9,347.18 |
| Hybrid Mamba 31-step | 88 | -7,438.48 | -0.047 | -0.689 | -10,292.62 |
| Mask-only hybrid Mamba | 88 | -8,020.68 | -0.050 | -0.743 | -10,147.87 |

The C2O proxy rows are diagnostic and have
`pnl_headline_eligible=false`. The V1 proxy-PnL headline remains `day_c2c`.

## Conservative Results

The current proxy evidence supports three narrow statements:

- Nonlinear tabular models, especially XGBoost and LightGBM, improve
  `jump_c2o` ranking metrics relative to the market-implied IVAR baseline and
  simple historical baselines.
- The strongest tabular models also improve the premium-space `day_c2c` proxy
  strategy screen after the current proxy haircut cost model.
- The sequence-model route is implemented, including daily, hybrid,
  intraday-only, and mask-only variants, but it is not the current headline
  result because coverage selection risk remains high and economic results are
  weak.

The defensible near-term claim is:

> In a no-NBBO proxy sample, state and event-history features show preliminary
> cross-sectional ranking signal for earnings event-variance mispricing beyond
> the market-implied IVAR baseline, and this ranking signal maps into positive
> premium-space proxy economics for the best tabular models.

## Limitations

These results should not be described as paper-grade executable performance.
The current route has the following limits:

- There are no historical bid/ask, quote midpoint, OPRA, or NBBO records.
- One-second option aggregates are trade OHLCV bars, so all IV surfaces and
  strategy marks are trade-price proxies.
- The sample begins in 2022 because the observed options day-aggregate
  entitlement does not cover the 2013-2025 target paper window.
- IVAR coverage is incomplete: 117 of 810 events lack a usable trade-proxy IVAR.
- Mamba sequence coverage is incomplete: 678 of 810 events pass the default
  sequence rule, a 16.3% drop rate.
- C2O and O2C option-PnL rows are diagnostic decompositions, not V1 tradable
  mispricing headlines.
- Cost sensitivity uses a proxy haircut, not full bid/ask crossing.

Paper-grade claims require historical quote/NBBO or equivalent data, quote-based
IVAR, leg-level execution with realistic bid/ask crossing, DTE and liquidity
robustness, and clustered or bootstrap inference.

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

QLIKE contribution diagnostic:

![QLIKE contribution diagnostic](assets/images/modeling/qlike_contribution_diagnostic.png)

## Artifact Map

Local raw outputs:

| Purpose | Path |
| --- | --- |
| Data pipeline manifest | `artifacts/data_pipeline/data_pipeline_manifest.json` |
| Dynamic calendar report | `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json` |
| Trade-proxy panel report | `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json` |
| Feature matrix | `data/gold/modeling/feature_matrix.parquet` |
| Daily sequence tensor | `data/gold/modeling/sequence_tensor.npz` |
| Hybrid sequence tensor | `data/gold/modeling/hybrid_sequence_tensor.npz` |
| Forecast metrics | `artifacts/modeling/forecast_metrics.csv` |
| Ranking metrics | `artifacts/modeling/ranking_metrics.csv` |
| Strategy metrics | `artifacts/modeling/strategy_metrics.csv` |
| Cost sensitivity | `artifacts/modeling/cost_sensitivity.csv` |
| Model diagnostics | `artifacts/modeling/model_fit_diagnostics.csv` |
| Proxy report | `reports/modeling/proxy_research_report.md` |

Published docs assets:

| Purpose | Path |
| --- | --- |
| Curated results page | `docs/results_snapshot.md` |
| Published figure copies | `docs/assets/images/modeling/*.png` |

