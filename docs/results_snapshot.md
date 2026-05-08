---
hide:
  - navigation
---

# Results Snapshot

This page is the paper-facing results snapshot for the current local proxy run.
It summarizes the data, model comparison, sequence diagnostics, robustness
checks, figures, and claim boundaries. Raw generated outputs remain under
ignored `artifacts/`, `data/`, and `reports/` paths; selected figures are
copied into `docs/assets/images/modeling/`.

The numeric tables below were cross-checked against the latest V5 Phase 1
artifact CSVs in `artifacts/modeling/`, including forecast, ranking, strategy,
sequence-gate, stacking, cost-sensitivity, and inference outputs.

## Research Question

The paper-facing question is:

> Can models improve trading decisions around option-implied earnings event
> variance mispricing?

This is not generic implied-volatility forecasting. Models forecast
`RVAR_event`; ex post mispricing is `RVAR_event - IVAR_event`; the trade layer
uses premium-space expected edge rather than raw variance edge alone.

The current run is a **no-NBBO proxy study**. It is suitable for a conservative
draft of the empirical story, but not for claims about executable bid/ask or
NBBO performance.

## Sample and Protocol

| Item | Current state |
| --- | --- |
| Verified local run | 2026-05-08 V5 Phase 1 proxy package |
| Study window | 2022-12-01 to 2025-12-31 |
| Target paper window | 2013-2025, pending historical quote/NBBO or equivalent data |
| Universe | Monthly top 50 liquid U.S. single-name option underlyings |
| Main timing sample | BMO and AMC earnings announcements |
| Event source | SEC EDGAR submissions plus SEC primary filing document text |
| Execution grade | `no_nbbo_trade_proxy`, `paper_grade=false` |
| Primary model split | Chronological 70/15/15 proxy split |
| Active sequence suite | V5 Phase 1 diagnostic suite |

### Data Route

| Data component | Role |
| --- | --- |
| Options day aggregates | Dynamic-universe liquidity ranking, contract discovery, local IV/IVAR proxy inputs, fallback diagnostics, and daily close-trade-implied option-surface sequences |
| Underlying stock day aggregates | Underlying closes, vendor OHLC opens, C2O/C2C/O2C event returns, and exit spot |
| Option one-second trade aggregates | Targeted pre-cutoff entry prices, primary C2C exit prices, and post-open C2O/O2C anchor prices |

The option one-second aggregates are trade OHLCV bars. They are not quote
midpoints, bid/ask records, OPRA, or NBBO.

### Target Definitions

| Target group | Model target | Current status | Interpretation |
| --- | --- | --- | --- |
| C2C | `day_c2c` | Modeled in V5 Phase 1 | Literature-compatible event-day variance and the only current proxy-PnL headline |
| C2O | `jump_c2o` | Modeled in V5 Phase 1 | Primary scientific ranking target for overnight earnings jumps |
| O2C | `reaction_o2c` | Not modeled in V5 Phase 1 | Post-open digestion diagnostic; reserved for Phase 2 only if sequence diagnostics justify expansion |

## Data Coverage

### Event and Contract Coverage

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

### IVAR Failure Diagnostics

| Failure reason | Events |
| --- | ---: |
| No two event-covering expiries | 103 |
| Nonmonotone total variance | 7 |
| Negative extracted IVAR | 7 |

The event panel is large enough for proxy-stage model comparison, but IVAR
coverage remains a material screen. The main IVAR loss channel is missing two
event-covering expiries.

### Sequence Coverage

| Measure | Value |
| --- | ---: |
| Total events | 810 |
| Daily sequence eligible events | 678 |
| Sequence drop rate | 16.3% |
| Events with at least eight valid intraday bins | 682 |
| Median hybrid feature-mask density | 0.7419 |
| Hybrid sequence too sparse | `false` |
| High sequence-selection risk | `true` |

The active hybrid tensor is `31 x 21`: 19 prior daily proxy-surface states plus
12 entry-day five-minute trade-aggregate proxy bins. Because the sequence drop
rate exceeds 10%, all sequence results remain diagnostic.

## Model Matrix

| Family | Models | Purpose |
| --- | --- | --- |
| Market benchmark | Market-implied `IVAR_event` | Central level and no-edge baseline |
| Historical baselines | Last-four RVAR, last-four IVAR | Tests whether simple earnings history is enough |
| Classical mispricing benchmark | Goyal-Saretto-style RV-IV spread | Required option-return predictability comparator |
| Linear tabular | Elastic Net | Sparse linear event-level benchmark |
| Nonlinear tabular | LightGBM, XGBoost | Main current contenders |
| Ensemble | LightGBM/XGBoost rank-average | Untuned equal-weight tree ensemble |
| Neural tabular | FT-Transformer | Deep tabular comparator |
| Sequence diagnostics | Ridge-flat aggregates, BiGRU, official bidirectional `mamba-ssm`, mask-only, time-shuffle | Tests whether ordered pre-event proxy-surface paths add value |

The legacy in-repo proxy-Mamba rows are retired because they used a gated
recurrent encoder rather than official `mamba-ssm`. They are retained only in
the machine-readable retirement manifest and cleanup tests.

## C2C Results

The C2C target is the current proxy-PnL headline because it has the cleanest
entry-to-exit premium-space proxy. The main empirical result is that nonlinear
tabular models dominate simple history and classical spread baselines in both
ranking and proxy economics.

### C2C Forecast and Ranking

| Model | N | RMSE | MAE | OOS R2 vs IVAR | AUC | Top-decile precision | Edge-decile Spearman |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Market IVAR | 100 | 0.0165 | 0.0105 | 0.000 | 0.500 | 0.000 | 0.079 |
| Last-four RVAR | 122 | 0.0362 | 0.0168 | -2.248 | 0.571 | 0.400 | 0.139 |
| Last-four IVAR | 122 | 0.0557 | 0.0190 | -0.278 | 0.495 | 0.300 | -0.139 |
| Goyal-Saretto spread | 100 | 0.0167 | 0.0101 | -0.019 | 0.621 | 0.500 | 0.503 |
| Elastic Net | 122 | 0.0272 | 0.0118 | 0.372 | 0.812 | 0.900 | 0.988 |
| LightGBM | 122 | 0.0200 | 0.0074 | 0.648 | 0.965 | 1.000 | 1.000 |
| XGBoost | 122 | 0.0211 | 0.0069 | 0.296 | 0.956 | 1.000 | 0.867 |
| LightGBM/XGBoost ensemble | 122 | 0.0199 | 0.0067 | 0.572 | 0.978 | 1.000 | 0.988 |
| FT-Transformer | 122 | 0.0408 | 0.0367 | -4.010 | 0.520 | 0.300 | 0.818 |
| Ridge-flat sequence aggregates | 122 | 0.0251 | 0.0123 | 0.264 | 0.636 | 0.500 | 0.758 |
| BiGRU sequence | 100 | 0.0294 | 0.0111 | -0.163 | 0.541 | 0.300 | 0.636 |
| Official `mamba-ssm` sequence | 100 | 0.0282 | 0.0108 | -0.016 | 0.504 | 0.300 | 0.406 |
| Mask-only sequence | 100 | 0.0295 | 0.0112 | -0.167 | 0.501 | 0.300 | 0.661 |
| Time-shuffle sequence | 100 | 0.0285 | 0.0107 | 0.020 | 0.532 | 0.200 | 0.697 |

### C2C Premium-Space Proxy Strategy

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|---:|---:|---:|---:|---:|
| Market IVAR | 0 | n/a | n/a | n/a | n/a |
| Last-four RVAR | 100 | -3,482 | -0.021 | -0.268 | -16,789 |
| Last-four IVAR | 100 | -15,904 | -0.094 | -1.235 | -16,132 |
| Goyal-Saretto spread | 100 | -461 | -0.003 | -0.036 | -17,145 |
| Elastic Net | 100 | 47,938 | 0.283 | 4.007 | -3,202 |
| LightGBM | 100 | 69,908 | 0.413 | 6.476 | -1,416 |
| XGBoost | 100 | 68,344 | 0.403 | 6.271 | -1,695 |
| LightGBM/XGBoost ensemble | 100 | 72,155 | 0.426 | 6.780 | -1,339 |
| FT-Transformer | 100 | -4,793 | -0.028 | -0.370 | -15,366 |
| Ridge-flat sequence aggregates | 100 | 19,918 | 0.118 | 1.557 | -7,338 |
| BiGRU sequence | 93 | 1,311 | 0.008 | 0.103 | -11,928 |
| Official `mamba-ssm` sequence | 93 | -4,178 | -0.026 | -0.329 | -13,765 |
| Mask-only sequence | 93 | 532 | 0.003 | 0.042 | -11,497 |
| Time-shuffle sequence | 93 | -20 | -0.000 | -0.002 | -11,928 |

**C2C interpretation.** The sellable result is the tabular nonlinear ranking
story. The LightGBM/XGBoost ensemble has the strongest C2C AUC and the highest
net proxy PnL. Sequence models do not improve the C2C headline; the official
`mamba-ssm` row is worse than simple sequence controls.

## C2O Results

The C2O target is the primary scientific ranking target because it isolates the
overnight earnings jump. Its option-PnL rows are diagnostic because they rely on
post-open trade-aggregate proxy anchors rather than a locked executable C2O
quote path.

### C2O Forecast and Ranking

| Model | N | RMSE | MAE | OOS R2 vs IVAR | AUC | Top-decile precision | Edge-decile Spearman |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Market IVAR | 100 | 0.0145 | 0.0097 | 0.000 | 0.500 | 0.000 | -0.006 |
| Last-four RVAR | 122 | 0.0293 | 0.0123 | -1.787 | 0.505 | 0.200 | 0.055 |
| Last-four IVAR | 122 | 0.0540 | 0.0181 | -0.295 | 0.468 | 0.200 | -0.333 |
| Goyal-Saretto spread | 100 | 0.0134 | 0.0076 | 0.141 | 0.602 | 0.300 | 0.685 |
| Elastic Net | 122 | 0.0213 | 0.0095 | 0.323 | 0.629 | 0.500 | 0.855 |
| LightGBM | 122 | 0.0192 | 0.0077 | 0.355 | 0.745 | 0.500 | 0.903 |
| XGBoost | 122 | 0.0191 | 0.0074 | 0.380 | 0.781 | 0.500 | 0.927 |
| LightGBM/XGBoost ensemble | 122 | 0.0191 | 0.0074 | 0.397 | 0.788 | 0.500 | 1.000 |
| FT-Transformer | 122 | 0.0396 | 0.0374 | -5.487 | 0.525 | 0.200 | 0.758 |
| Ridge-flat sequence aggregates | 122 | 0.0221 | 0.0110 | -0.047 | 0.434 | 0.200 | -0.030 |
| BiGRU sequence | 100 | 0.0236 | 0.0085 | 0.057 | 0.496 | 0.200 | 0.867 |
| Official `mamba-ssm` sequence | 100 | 0.0237 | 0.0087 | 0.042 | 0.495 | 0.200 | 0.527 |
| Mask-only sequence | 100 | 0.0242 | 0.0088 | -0.034 | 0.500 | 0.100 | 0.685 |
| Time-shuffle sequence | 100 | 0.0248 | 0.0092 | -0.202 | 0.475 | 0.100 | 0.915 |

### C2O Diagnostic Strategy Proxies

| Proxy exit | Best model | Trades | Net proxy PnL | Return on premium | Sharpe | Headline eligible |
|:---|:---|---:|---:|---:|---:|:---|
| Intrinsic open diagnostic | LightGBM/XGBoost ensemble | 100 | 53,962 | 0.319 | 4.470 | false |
| Post-open option VWAP 0-5m | XGBoost | 95 | 40,512 | 0.239 | 4.347 | false |
| Post-open option VWAP 5-15m | XGBoost | 93 | 41,456 | 0.245 | 4.190 | false |

Selected 5-15 minute post-open option-VWAP proxy diagnostics:

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|---:|---:|---:|---:|---:|
| Last-four RVAR | 93 | -805 | -0.005 | -0.074 | -10,017 |
| Goyal-Saretto spread | 93 | -1,531 | -0.009 | -0.141 | -12,854 |
| Elastic Net | 93 | 14,324 | 0.085 | 1.335 | -5,844 |
| LightGBM | 93 | 28,911 | 0.171 | 2.782 | -3,910 |
| XGBoost | 93 | 41,456 | 0.245 | 4.190 | -1,698 |
| FT-Transformer | 93 | 4,113 | 0.024 | 0.380 | -9,347 |
| Official `mamba-ssm` sequence | 88 | -8,668 | -0.055 | -0.804 | -11,086 |

**C2O interpretation.** XGBoost and the tree ensemble are the strongest
scientific ranking models. The C2O strategy proxies are directionally
consistent with the tabular story but remain diagnostic because the exit anchor
is post-open trade-aggregate VWAP rather than an executable quote-based mark.

## O2C Diagnostic Results

No `reaction_o2c` model table is present in the V5 Phase 1 artifacts. This is
intentional: Phase 1 runs only `jump_c2o` and `day_c2c`; `reaction_o2c` is
reserved for Phase 2 only if a real sequence model passes the diagnostic gate.

The current O2C evidence is therefore limited to trade-proxy decomposition:

| O2C diagnostic | Value |
| --- | ---: |
| Mean O2C option VWAP 5-15m to primary C2C exit diagnostic PnL | -24.31 USD |
| Mean option-proxy decomposition residual, 5-15m | 0.00 USD |

**O2C interpretation.** The current data support an O2C decomposition check, not
an O2C model claim. This should be written as a diagnostic appendix item unless
Phase 2 is reopened with broader sequence evidence.

## Sequence and Mamba Diagnostics

The Phase 1 sequence suite is a diagnostic-grade signal test, not a headline
model run. The gate requires:

- AUC lift of at least 0.05.
- Bootstrap lift 95% CI lower bound above zero.
- Same target, split, and common-row universe.
- No sequence-selection headline override when coverage risk is high.

No sequence model passes this gate in the current sample.

| Target | Model | Test N | AUC | AUC lift | 95% CI low | 95% CI high | Gate |
|:---|:---|---:|---:|---:|---:|---:|:---|
| `jump_c2o` | BiGRU | 93 | 0.496 | -0.004 vs mask-only | -0.028 | 0.026 | fail |
| `jump_c2o` | Official `mamba-ssm` | 93 | 0.495 | -0.005 vs mask-only | -0.063 | 0.050 | fail |
| `jump_c2o` | Ridge-flat aggregates | 100 | 0.434 | -0.066 vs mask-only | -0.206 | 0.017 | fail |
| `day_c2c` | Ridge-flat aggregates | 100 | 0.636 | 0.104 vs time-shuffle | -0.044 | 0.282 | fail |
| `day_c2c` | BiGRU | 93 | 0.541 | 0.009 vs time-shuffle | -0.009 | 0.029 | fail |
| `day_c2c` | Official `mamba-ssm` | 93 | 0.504 | -0.027 vs time-shuffle | -0.099 | 0.037 | fail |

Cross-fitted stacking into the LightGBM/XGBoost ensemble is also negative on
the locked test sample:

| Target | Sequence model | Test N | Baseline AUC | Stacked AUC | AUC lift |
|:---|:---|---:|---:|---:|---:|
| `jump_c2o` | Ridge-flat aggregates | 100 | 0.788 | 0.657 | -0.131 |
| `jump_c2o` | Official `mamba-ssm` | 93 | 0.786 | 0.572 | -0.213 |
| `jump_c2o` | BiGRU | 93 | 0.786 | 0.463 | -0.322 |
| `day_c2c` | Ridge-flat aggregates | 100 | 0.978 | 0.958 | -0.019 |
| `day_c2c` | BiGRU | 93 | 0.976 | 0.895 | -0.081 |
| `day_c2c` | Official `mamba-ssm` | 93 | 0.976 | 0.882 | -0.094 |

**Sequence interpretation.** Ordered proxy-surface paths do not provide
reliable incremental information in the current sample. The official
`mamba-ssm` implementation runs on CUDA and is now a valid challenger, but it
does not beat the gate, the controls, or the tabular ensemble.

## Robustness and Inference

### Cost Stress

Selected `day_c2c` net proxy PnL under entry-premium haircut multipliers:

| Model | 1x cost | 3x cost | 5x cost |
|:---|---:|---:|---:|
| Goyal-Saretto spread | -461 | -2,155 | -3,849 |
| Elastic Net | 47,938 | 46,244 | 44,550 |
| LightGBM | 69,908 | 68,214 | 66,520 |
| XGBoost | 68,344 | 66,650 | 64,956 |
| LightGBM/XGBoost ensemble | 72,155 | 70,461 | 68,767 |

Cost stress supports the same conclusion as the main C2C strategy table:
tabular nonlinear ranking remains economically meaningful under the current
proxy haircut model.

### Forecast-Loss Inference

Positive values indicate lower squared forecast loss than market IVAR. The
common 100-row test set has 46 event-date clusters and 72 ticker clusters.

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

Clustered forecast-loss diagnostics are strongest for LightGBM in the current
proxy run. Ranking and premium-space economics remain the primary evidence;
lower RMSE alone is not sufficient for the paper claim.

### QLIKE and Calibration Caveat

QLIKE diagnostics are retained as sanity checks but are not used as the primary
claim surface because very small realized variance values can dominate the
raw QLIKE contribution. The paper-facing argument should emphasize ranking,
edge-decile monotonicity, cost sensitivity, and premium-space proxy economics.

## Figures

All generated paper-facing figures from `reports/modeling/figures/*.png` are
synced into `docs/assets/images/modeling/` and included below.

### Forecast Error

![Forecast performance](assets/images/modeling/forecast_performance.png)

### Ranking Quality

![AUC and top-decile precision](assets/images/modeling/auc_top_decile_precision.png)

### Premium-Space Proxy Economics

![Strategy PnL by edge decile](assets/images/modeling/strategy_pnl_by_edge_decile.png)

### Cost Sensitivity

![Cost sensitivity](assets/images/modeling/cost_sensitivity.png)

### Calibration

![Calibration plot](assets/images/modeling/calibration_plot.png)

### Realized Mispricing by Predicted Edge Decile

![Edge decile realized mispricing](assets/images/modeling/edge_decile_realized_mispricing.png)

### QLIKE Contribution Diagnostic

![QLIKE contribution diagnostic](assets/images/modeling/qlike_contribution_diagnostic.png)

## Discussion

### What the Results Support

The current proxy evidence supports a conservative tabular-model contribution:

1. Nonlinear tabular models improve earnings-event mispricing ranking relative
   to market IVAR, simple historical baselines, and the classical RV-IV spread.
2. The strongest models also improve the C2C premium-space proxy strategy after
   the current cost haircut.
3. C2O rankings and diagnostic post-open option-VWAP proxies point in the same
   direction, but remain diagnostic rather than headline tradability evidence.
4. Sequence diagnostics are informative as a negative result: at current proxy
   coverage and quality, ordered pre-event proxy-surface paths do not beat
   tabular aggregates or controls.

### What the Results Do Not Support

The current run does not support:

- a Mamba headline;
- a general sequence-model headline;
- a claim of NBBO-executable performance;
- full-spread tradability;
- a paper claim based only on lower RMSE;
- a modeled `reaction_o2c` result.

### Sellable Claim

The defensible near-term claim is:

> In a no-NBBO proxy sample, nonlinear tabular models identify cross-sectional
> earnings event-variance mispricing beyond market IVAR and classical
> history/spread baselines, and the ranking signal maps into positive
> premium-space proxy economics for the best tree models. Sequence/Mamba
> models were audited with strong controls and do not currently provide
> reliable incremental value.

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

Paper-grade claims require historical quote/NBBO or equivalent data,
quote-based IVAR, leg-level execution with realistic bid/ask crossing, DTE and
liquidity robustness, and clustered or bootstrap inference.

## Artifact Map

### Raw Outputs

| Purpose | Path |
| --- | --- |
| Data pipeline manifest | `artifacts/data_pipeline/data_pipeline_manifest.json` |
| Dynamic calendar report | `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json` |
| Trade-proxy panel report | `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json` |
| Feature matrix | `data/gold/modeling/feature_matrix.parquet` |
| Daily sequence tensor | `data/gold/modeling/sequence_tensor.npz` |
| Hybrid sequence tensor V1 | `data/gold/modeling/hybrid_sequence_tensor.npz` |
| Hybrid sequence tensor V2 | `data/gold/modeling/hybrid_sequence_tensor_v2.npz` |
| Forecast metrics | `artifacts/modeling/forecast_metrics.csv` |
| Ranking metrics | `artifacts/modeling/ranking_metrics.csv` |
| Strategy metrics | `artifacts/modeling/strategy_metrics.csv` |
| Cost sensitivity | `artifacts/modeling/cost_sensitivity.csv` |
| Forecast-loss inference | `artifacts/modeling/inference.csv` |
| Common-row pairwise metrics | `artifacts/modeling/common_row_pairwise_metrics.csv` |
| Clustered bootstrap CI | `artifacts/modeling/clustered_bootstrap_ci.csv` |
| Incremental value diagnostics | `artifacts/modeling/incremental_value_diagnostics.csv` |
| Sequence model gate diagnostics | `artifacts/modeling/sequence_model_fit_diagnostics.csv` |
| Sequence quality diagnostics | `artifacts/modeling/sequence_v2_quality.csv` |
| Model diagnostics | `artifacts/modeling/model_fit_diagnostics.csv` |
| Retired model manifest | `artifacts/modeling/retired_model_ids.json` |
| Proxy report | `reports/modeling/proxy_research_report.md` |

### Published Docs Assets

| Purpose | Path |
| --- | --- |
| Curated results page | `docs/results_snapshot.md` |
| Forecast figure | `docs/assets/images/modeling/forecast_performance.png` |
| Ranking figure | `docs/assets/images/modeling/auc_top_decile_precision.png` |
| Strategy figure | `docs/assets/images/modeling/strategy_pnl_by_edge_decile.png` |
| Cost sensitivity figure | `docs/assets/images/modeling/cost_sensitivity.png` |
| Calibration figure | `docs/assets/images/modeling/calibration_plot.png` |
| Edge-decile realized mispricing figure | `docs/assets/images/modeling/edge_decile_realized_mispricing.png` |
| QLIKE contribution diagnostic figure | `docs/assets/images/modeling/qlike_contribution_diagnostic.png` |
