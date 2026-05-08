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
rows, validates on 121 rows, and tests on 122 rows for tabular models. The
legacy in-repo proxy-Mamba rows are retired because they used a gated recurrent
encoder rather than official `mamba-ssm`. The new sequence suite is a
diagnostic-grade test of whether ordered pre-event proxy-surface paths add
incremental information beyond tabular aggregates.

Selected `jump_c2o` forecast metrics:

| Model | N | MAE | RMSE | OOS R2 vs IVAR |
|:---|---:|---:|---:|---:|
| Market IVAR | 100 | 0.0097 | 0.0145 | 0.000 |
| Goyal-Saretto spread | 100 | 0.0076 | 0.0134 | 0.141 |
| Elastic Net | 122 | 0.0095 | 0.0213 | 0.323 |
| LightGBM | 122 | 0.0077 | 0.0192 | 0.355 |
| XGBoost | 122 | 0.0074 | 0.0191 | 0.380 |
| FT-Transformer | 122 | 0.0374 | 0.0396 | -5.487 |

Selected `jump_c2o` ranking metrics:

| Model | N | Top-decile precision | AUC | Edge-decile Spearman |
|:---|---:|---:|---:|---:|
| Market IVAR | 100 | 0.000 | 0.500 | -0.006 |
| Goyal-Saretto spread | 100 | 0.300 | 0.602 | 0.685 |
| Elastic Net | 100 | 0.500 | 0.629 | 0.855 |
| LightGBM | 100 | 0.500 | 0.745 | 0.903 |
| XGBoost | 100 | 0.500 | 0.781 | 0.927 |
| FT-Transformer | 100 | 0.200 | 0.525 | 0.758 |

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

Selected `jump_c2o` 5-15 minute post-open option-VWAP proxy diagnostics:

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|---:|---:|---:|---:|---:|
| Last-four RVAR | 93 | -804.97 | -0.005 | -0.074 | -10,016.79 |
| Goyal-Saretto spread | 93 | -1,531.36 | -0.009 | -0.141 | -12,854.27 |
| Elastic Net | 93 | 14,324.21 | 0.085 | 1.335 | -5,844.42 |
| LightGBM | 93 | 28,911.28 | 0.171 | 2.782 | -3,909.83 |
| XGBoost | 93 | 41,455.71 | 0.245 | 4.190 | -1,697.70 |
| FT-Transformer | 93 | 4,112.63 | 0.024 | 0.380 | -9,347.18 |

The C2O proxy rows are diagnostic and have
`pnl_headline_eligible=false`. The V1 proxy-PnL headline remains `day_c2c`.

### Phase 1 Sequence Gate

The Phase 1 sequence suite is a diagnostic-grade signal test, not a headline
model run. The gate requires AUC lift of at least 0.05 and a bootstrap lift
95% CI with lower bound above zero on the same common-row universe. No sequence
model passes this gate in the current sample, and sequence stacking reduces the
locked-test AUC versus the LightGBM/XGBoost rank-average ensemble.

Selected common-row Phase 1 gate results:

| Target | Model | Test N | AUC | AUC lift vs control | 95% CI low | 95% CI high | Gate |
|:---|:---|---:|---:|---:|---:|---:|:---|
| `jump_c2o` | BiGRU | 93 | 0.496 | -0.004 vs mask-only | -0.028 | 0.026 | fail |
| `jump_c2o` | official `mamba-ssm` | 93 | 0.495 | -0.005 vs mask-only | -0.063 | 0.050 | fail |
| `jump_c2o` | ridge-flat aggregates | 100 | 0.434 | -0.066 vs mask-only | -0.206 | 0.017 | fail |
| `day_c2c` | ridge-flat aggregates | 100 | 0.636 | 0.104 vs time-shuffle | -0.044 | 0.282 | fail |
| `day_c2c` | BiGRU | 93 | 0.541 | 0.009 vs time-shuffle | -0.009 | 0.029 | fail |
| `day_c2c` | official `mamba-ssm` | 93 | 0.504 | -0.027 vs time-shuffle | -0.099 | 0.037 | fail |

Interpretation: ordered proxy-surface paths do not provide reliable
incremental signal in the current sample. Official `mamba-ssm` is implemented
and runs on CUDA, but it is not a paper headline model.

## Robustness and Inference

The current robustness package is still proxy-stage, but it is now enough for a
conservative manuscript draft to discuss cost stress and clustered forecast-loss
diagnostics. The robustness tables focus on the classical benchmark and tabular
contenders because those are the only current sellable models; sequence models
remain diagnostic until sequence-selection risk is reduced.

Selected `day_c2c` net proxy PnL under entry-premium haircut multipliers:

| Model | 1x cost | 3x cost | 5x cost |
|:---|---:|---:|---:|
| Goyal-Saretto spread | -461 | -2,155 | -3,849 |
| Elastic Net | 47,938 | 46,244 | 44,550 |
| LightGBM | 69,908 | 68,214 | 66,520 |
| XGBoost | 68,344 | 66,650 | 64,956 |

Forecast-loss inference uses the test rows and reports the mean squared-loss
improvement versus market IVAR. Positive values mean lower squared forecast
loss than IVAR. The current table uses 46 event-date clusters and 72 ticker
clusters for the common 100-row test set.

Selected two-way clustered forecast-loss diagnostics:

| Target | Model | Mean loss improvement vs IVAR | Two-way cluster SE | Ratio |
|:---|:---|---:|---:|---:|
| `jump_c2o` | Goyal-Saretto spread | 0.000030 | 0.000022 | 1.35 |
| `jump_c2o` | Elastic Net | 0.000068 | 0.000042 | 1.60 |
| `jump_c2o` | LightGBM | 0.000074 | 0.000034 | 2.19 |
| `jump_c2o` | XGBoost | 0.000080 | 0.000043 | 1.87 |
| `day_c2c` | Goyal-Saretto spread | -0.000005 | 0.000039 | -0.13 |
| `day_c2c` | Elastic Net | 0.000101 | 0.000051 | 1.99 |
| `day_c2c` | LightGBM | 0.000177 | 0.000070 | 2.51 |
| `day_c2c` | XGBoost | 0.000081 | 0.000108 | 0.75 |

Interpretation: cost stress supports the same tabular ranking story as the
headline proxy strategy table. Clustered forecast-loss diagnostics are strongest
for LightGBM in the current proxy run, with XGBoost still leading the
`jump_c2o` ranking metrics. These diagnostics are not a substitute for
paper-grade execution inference under bid/ask or NBBO data.

## Conservative Results

The current proxy evidence supports three narrow statements:

- Nonlinear tabular models, especially XGBoost and LightGBM, improve
  `jump_c2o` ranking metrics relative to the market-implied IVAR baseline and
  simple historical baselines.
- The strongest tabular models also improve the premium-space `day_c2c` proxy
  strategy screen after the current proxy haircut cost model.
- The new sequence-model route is diagnostic-grade. It compares ridge-flat
  sequence aggregates, BiGRU, official `mamba-ssm`, mask-only, and time-shuffle
  controls before any sequence claim can enter the paper headline.

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
- Sequence coverage is incomplete: 678 of 810 events pass the default sequence
  rule, a 16.3% drop rate.
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
