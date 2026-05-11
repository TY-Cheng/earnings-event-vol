---
hide:
  - navigation
---

# Results Snapshot

This page is the current paper-facing result ledger for the local
`tuned_phase1` proxy run. Raw generated outputs remain under ignored
`artifacts/`, `reports/`, and external `DATA_DIR` locations; selected figures
are copied into `docs/assets/images/modeling/`.

## Current Verdict

The repo now has the full `tuned_phase1` comparison package for the available
proxy data: data panel, model implementations, tuned and untuned rows, figures,
tables, and current analysis notes. It is good enough for a conservative
working paper draft and internal research review.

It is not yet good enough for paper-grade execution claims. All current option
prices are trade-aggregate proxies, not bid/ask, OPRA, or NBBO. The current
execution grade remains `no_nbbo_trade_proxy`; `paper_grade=false`.

## Research Question

The paper-facing question is:

> Can models improve trading decisions around option-implied earnings event
> variance mispricing?

This is not generic implied-volatility forecasting. Models forecast
`RVAR_event`; ex post mispricing is `RVAR_event - IVAR_event`; the trade layer
uses premium-space expected edge rather than raw variance edge alone.

The target system is:

| Target | Role | Definition |
| --- | --- | --- |
| `jump_c2o` | Primary scientific ranking target | close-to-open earnings jump variance |
| `day_c2c` | V1 proxy-PnL headline | close-to-close full reaction-day variance |
| `reaction_o2c` | Diagnostic target | open-to-close post-open digestion variance |

## Sample and Protocol

| Item | Current state |
| --- | --- |
| Verified local run | 2026-05-11 `tuned_phase1` proxy package |
| Command | `just research args="--stage all --sequence-suite phase1 --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1"` |
| Study window | 2022-12-01 to 2025-12-31 |
| Target paper window | 2013-2025, pending historical quote/NBBO or equivalent data |
| Universe | Monthly top 50 liquid U.S. single-name option underlyings |
| Main timing sample | BMO and AMC earnings announcements |
| Event source | SEC EDGAR submissions plus SEC primary filing document text |
| Execution grade | `no_nbbo_trade_proxy`, `paper_grade=false` |
| Split | Chronological event-level 70/15/15 |
| Tuning profile | `tuned_phase1`, explicit only; `untuned` remains the default |
| Tuning seed | 17 |
| Bootstrap iterations | 1,000 |
| Research manifest | `artifacts/modeling/research_manifest.json`, `ok=true` |

## Data Coverage

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

Sequence coverage:

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
rate exceeds 10%, sequence rows remain diagnostic.

## Model Matrix

| Family | Models | Status |
| --- | --- | --- |
| Market benchmark | Market-implied `IVAR_event` | Active neutral edge baseline |
| Historical baselines | Last-four RVAR, last-four IVAR | Active deterministic baselines |
| Classical spread benchmark | Goyal-Saretto-style RV-IV spread | Active option-return predictability comparator |
| Linear tabular | Elastic Net, Elastic Net tuned | Active; untuned is repo coordinate descent, tuned is sklearn `ElasticNetCV` |
| Nonlinear tabular | LightGBM, LightGBM tuned, XGBoost, XGBoost tuned | Active; tuned rows use validation-only selection and train+validation refit |
| Ensemble | LightGBM/XGBoost rank-average | Active untuned equal-weight rank ensemble |
| Neural tabular | FT-Transformer, FT-Transformer tuned | Untuned FT is invalid/no usable predictions; tuned FT trains but is not competitive |
| Sequence diagnostics | Ridge-flat aggregates, BiGRU, BiGRU 5-seed, official `mamba-ssm`, official `mamba-ssm` 5-seed, mask-only, time-shuffle | Diagnostic only; no sequence row passes the gate |

The `tuned_phase1` artifacts are:

- `artifacts/modeling/tuning_trials.csv`: 131 validation-only trial rows.
- `artifacts/modeling/tuning_selected_params.json`: selected tuned parameters;
  `test_metrics_used_for_selection=false`.
- `artifacts/modeling/model_fit_diagnostics.csv`: fit status, finite-prediction
  counts, tuning metadata, and sequence seed diagnostics.

The legacy in-repo proxy-Mamba ids
`daily_mamba_20step`, `hybrid_mamba_31step`,
`intraday_only_mamba_12step`, and `mask_only_hybrid_mamba` are retired because
they used a gated recurrent encoder rather than official `mamba-ssm`.

## Headline Result Summary

Forecast and ranking columns use `jump_c2o`, the primary scientific ranking
target. The strategy column uses `day_c2c`, the only current V1 proxy-PnL
headline. `n/a` for untuned FT-Transformer means no finite validation/test
predictions were produced; it is not a zero result.

| Model | MAE | RMSE | OOS R2 vs IVAR | Top-decile precision | AUC | Day-C2C net proxy PnL |
|:---|---:|---:|---:|---:|---:|---:|
| Market IVAR | 0.0097 | 0.0145 | 0.000 | 0.000 | 0.500 | n/a |
| Last-four RVAR | 0.0123 | 0.0293 | -1.787 | 0.200 | 0.505 | -3,482 |
| Last-four IVAR | 0.0181 | 0.0540 | -0.295 | 0.200 | 0.468 | -15,904 |
| Goyal-Saretto spread | 0.0076 | 0.0134 | 0.141 | 0.300 | 0.602 | -461 |
| Elastic Net | 0.0095 | 0.0213 | 0.323 | 0.500 | 0.629 | 47,938 |
| Elastic Net tuned | 0.0086 | 0.0200 | 0.372 | 0.600 | 0.644 | 48,770 |
| LightGBM | 0.0077 | 0.0192 | 0.355 | 0.500 | 0.745 | 69,908 |
| LightGBM tuned | 0.0085 | 0.0197 | 0.321 | 0.500 | 0.666 | 50,370 |
| XGBoost | 0.0074 | 0.0191 | 0.380 | 0.500 | 0.781 | 68,344 |
| XGBoost tuned | 0.0079 | 0.0191 | 0.415 | 0.500 | 0.714 | 59,497 |
| LightGBM/XGBoost ensemble | 0.0074 | 0.0191 | 0.397 | 0.500 | 0.788 | 72,155 |
| FT-Transformer | n/a | n/a | n/a | n/a | n/a | n/a |
| FT-Transformer tuned | 0.2111 | 0.2122 | -214.079 | 0.200 | 0.490 | -5,934 |
| BiGRU sequence | 0.0086 | 0.0235 | 0.078 | 0.100 | 0.471 | -6,078 |
| BiGRU 5-seed | 0.0087 | 0.0236 | 0.059 | 0.200 | 0.472 | -8,022 |
| Official `mamba-ssm` | 0.0087 | 0.0238 | 0.044 | 0.200 | 0.510 | -2,692 |
| Official `mamba-ssm` 5-seed | 0.0088 | 0.0239 | 0.024 | 0.200 | 0.501 | -1,793 |
| Mask-only sequence | 0.0088 | 0.0242 | -0.039 | 0.100 | 0.500 | 101 |
| Time-shuffle sequence | 0.0086 | 0.0237 | 0.056 | 0.200 | 0.475 | -8,022 |

**Interpretation.** The strongest sellable result remains tabular nonlinear
ranking, not deep sequence modeling. The LightGBM/XGBoost rank-average ensemble
leads `jump_c2o` AUC and `day_c2c` net proxy PnL. Tuning improves some
individual diagnostics, especially XGBoost OOS R2 and Elastic Net top-decile
precision, but it does not overturn the headline ensemble result.

## C2C Proxy Strategy

The C2C target is the current proxy-PnL headline because it has the cleanest
entry-to-exit premium-space proxy.

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|---:|---:|---:|---:|---:|
| Market IVAR | 0 | n/a | n/a | n/a | n/a |
| Last-four RVAR | 100 | -3,482 | -0.021 | -0.268 | -16,789 |
| Last-four IVAR | 100 | -15,904 | -0.094 | -1.235 | -16,132 |
| Goyal-Saretto spread | 100 | -461 | -0.003 | -0.036 | -17,145 |
| Elastic Net | 100 | 47,938 | 0.283 | 4.007 | -3,202 |
| Elastic Net tuned | 100 | 48,770 | 0.288 | 4.088 | -2,846 |
| LightGBM | 100 | 69,908 | 0.413 | 6.476 | -1,416 |
| LightGBM tuned | 100 | 50,370 | 0.297 | 4.245 | -4,346 |
| XGBoost | 100 | 68,344 | 0.403 | 6.271 | -1,695 |
| XGBoost tuned | 100 | 59,497 | 0.351 | 5.209 | -2,846 |
| LightGBM/XGBoost ensemble | 100 | 72,155 | 0.426 | 6.780 | -1,339 |
| FT-Transformer | 0 | n/a | n/a | n/a | n/a |
| FT-Transformer tuned | 100 | -5,934 | -0.035 | -0.458 | -16,507 |
| BiGRU sequence | 93 | -6,078 | -0.038 | -0.480 | -13,765 |
| BiGRU 5-seed | 93 | -8,022 | -0.050 | -0.634 | -15,709 |
| Official `mamba-ssm` | 93 | -2,692 | -0.017 | -0.212 | -13,038 |
| Official `mamba-ssm` 5-seed | 93 | -1,793 | -0.011 | -0.141 | -13,038 |
| Mask-only sequence | 93 | 101 | 0.001 | 0.008 | -11,928 |
| Time-shuffle sequence | 93 | -8,022 | -0.050 | -0.634 | -15,709 |

![Strategy PnL by edge decile](assets/images/modeling/strategy_pnl_by_edge_decile.png)

## C2O Ranking

The C2O target is the primary scientific ranking target because it isolates the
overnight earnings jump. Option-PnL rows for C2O remain diagnostic because they
use post-open trade-aggregate proxy anchors rather than a locked executable
C2O quote path.

| Model | N | RMSE | MAE | OOS R2 vs IVAR | AUC | Top-decile precision | Edge-decile Spearman |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Market IVAR | 100 | 0.0145 | 0.0097 | 0.000 | 0.500 | 0.000 | -0.006 |
| Goyal-Saretto spread | 100 | 0.0134 | 0.0076 | 0.141 | 0.602 | 0.300 | 0.685 |
| Elastic Net | 122 | 0.0213 | 0.0095 | 0.323 | 0.629 | 0.500 | 0.855 |
| Elastic Net tuned | 122 | 0.0200 | 0.0086 | 0.372 | 0.644 | 0.600 | 0.612 |
| LightGBM | 122 | 0.0192 | 0.0077 | 0.355 | 0.745 | 0.500 | 0.903 |
| LightGBM tuned | 122 | 0.0197 | 0.0085 | 0.321 | 0.666 | 0.500 | 0.855 |
| XGBoost | 122 | 0.0191 | 0.0074 | 0.380 | 0.781 | 0.500 | 0.927 |
| XGBoost tuned | 122 | 0.0191 | 0.0079 | 0.415 | 0.714 | 0.500 | 0.818 |
| LightGBM/XGBoost ensemble | 122 | 0.0191 | 0.0074 | 0.397 | 0.788 | 0.500 | 1.000 |
| FT-Transformer | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| FT-Transformer tuned | 122 | 0.2122 | 0.2111 | -214.079 | 0.490 | 0.200 | 0.636 |
| Official `mamba-ssm` | 100 | 0.0238 | 0.0087 | 0.044 | 0.510 | 0.200 | 0.685 |
| Official `mamba-ssm` 5-seed | 100 | 0.0239 | 0.0088 | 0.024 | 0.501 | 0.200 | 0.479 |

![AUC and top-decile precision](assets/images/modeling/auc_top_decile_precision.png)

## O2C Diagnostic

`reaction_o2c` is modeled as a diagnostic target. Its realized variance is
post-open only, while `IVAR_event` is a full-event implied-variance comparator.
That makes O2C useful for ranking and decomposition, not level-calibrated
mispricing claims.

| Model | N | RMSE | MAE | OOS R2 vs IVAR | AUC | Top-decile precision |
|:---|---:|---:|---:|---:|---:|---:|
| Market IVAR | 100 | 0.0141 | 0.0105 | 0.000 | 0.500 | 0.000 |
| Elastic Net | 122 | 0.0084 | 0.0029 | 0.944 | 0.695 | 0.200 |
| Elastic Net tuned | 122 | 0.0084 | 0.0030 | 0.944 | 0.698 | 0.200 |
| LightGBM | 122 | 0.0075 | 0.0030 | 0.926 | 0.818 | 0.300 |
| LightGBM tuned | 122 | 0.0079 | 0.0029 | 0.944 | 0.750 | 0.200 |
| XGBoost | 122 | 0.0080 | 0.0032 | 0.909 | 0.810 | 0.300 |
| XGBoost tuned | 122 | 0.0078 | 0.0027 | 0.951 | 0.768 | 0.200 |
| LightGBM/XGBoost ensemble | 122 | 0.0076 | 0.0029 | 0.927 | 0.823 | 0.300 |
| FT-Transformer | n/a | n/a | n/a | n/a | n/a | n/a |
| FT-Transformer tuned | 122 | 0.2163 | 0.2161 | -235.998 | 0.703 | 0.200 |
| Official `mamba-ssm` | 100 | 0.0038 | 0.0020 | 0.930 | 0.659 | 0.100 |

All O2C strategy rows are `pnl_headline_eligible=false`.

![O2C AUC and top-decile precision](assets/images/modeling/o2c_auc_top_decile_precision.png)

## Cost Sensitivity

| Model | 1x cost | 3x cost | 5x cost |
|:---|---:|---:|---:|
| Goyal-Saretto spread | -461 | -2,155 | -3,849 |
| Elastic Net | 47,938 | 46,244 | 44,550 |
| Elastic Net tuned | 48,770 | 47,076 | 45,382 |
| LightGBM | 69,908 | 68,214 | 66,520 |
| LightGBM tuned | 50,370 | 48,676 | 46,983 |
| XGBoost | 68,344 | 66,650 | 64,956 |
| XGBoost tuned | 59,497 | 57,804 | 56,110 |
| LightGBM/XGBoost ensemble | 72,155 | 70,461 | 68,767 |
| Official `mamba-ssm` | -2,692 | -4,281 | -5,870 |
| Official `mamba-ssm` 5-seed | -1,793 | -3,383 | -4,972 |

![Cost sensitivity](assets/images/modeling/cost_sensitivity.png)

## Sequence Diagnostics

No sequence model passes the diagnostic gate. The official `mamba-ssm` rows are
validly implemented diagnostics, but they do not beat tabular baselines or
controls robustly enough to upgrade the claim.

| Target | Model | Test N | AUC lift | 95% CI low | 95% CI high | Gate |
|:---|:---|---:|---:|---:|---:|:---|
| `jump_c2o` | BiGRU sequence | 93 | -0.029 | -0.070 | 0.008 | fail |
| `jump_c2o` | BiGRU 5-seed | 93 | -0.028 | -0.070 | 0.012 | fail |
| `jump_c2o` | Official `mamba-ssm` | 93 | 0.010 | -0.053 | 0.097 | fail |
| `jump_c2o` | Official `mamba-ssm` 5-seed | 93 | 0.001 | -0.065 | 0.091 | fail |
| `day_c2c` | BiGRU 5-seed | 93 | 0.025 | -0.034 | 0.092 | fail |
| `day_c2c` | Official `mamba-ssm` | 93 | -0.030 | -0.078 | 0.005 | fail |
| `day_c2c` | Official `mamba-ssm` 5-seed | 93 | -0.027 | -0.075 | 0.005 | fail |
| `reaction_o2c` | Official `mamba-ssm` | 93 | 0.059 | -0.017 | 0.207 | fail |

Cross-fitted stacking of sequence forecasts into the LightGBM/XGBoost ensemble
is negative on the locked test sample for the current proxy run.

## Figure Ledger

| Figure | Purpose |
| --- | --- |
| `forecast_performance.png` | Jump target forecast error and OOS R2 |
| `auc_top_decile_precision.png` | Jump target ranking and tail precision |
| `edge_decile_realized_mispricing.png` | Realized mispricing by predicted-edge decile |
| `strategy_pnl_by_edge_decile.png` | C2C proxy economics by edge decile |
| `cost_sensitivity.png` | Cost multiplier robustness |
| `qlike_contribution_diagnostic.png` | Forecast-loss diagnostic |
| `o2c_forecast_performance.png` | O2C diagnostic forecast performance |
| `o2c_auc_top_decile_precision.png` | O2C diagnostic ranking |
| `o2c_strategy_proxy_pnl.png` | O2C diagnostic proxy PnL |
| `o2c_scale_diagnostic.png` | O2C/full-event IVAR scale mismatch |

## What We Can Sell

The defensible near-term claim is:

> In a no-NBBO proxy sample, state and event-history features show preliminary
> cross-sectional ranking signal for earnings event-variance mispricing beyond
> the market-implied IVAR baseline, and the best tabular models map that signal
> into positive premium-space proxy economics.

The current paper angle should emphasize:

1. The trading question is event-variance mispricing, not generic IV
   forecasting.
2. Ranking and top-decile selection matter more than unconditional RMSE.
3. The tree ensemble wins the current proxy evidence; tuned rows are important
   fairness checks, not a replacement for the main ensemble result.
4. Sequence/Mamba is currently a negative diagnostic result.

## Claim Boundaries

Do not claim:

- paper-grade executable performance;
- bid/ask, OPRA, or NBBO execution;
- full-spread tradability;
- Mamba superiority;
- that lower RMSE alone implies economic value;
- that O2C rows are calibrated mispricing strategies.

Paper-grade claims require historical quote/NBBO or equivalent data,
quote-based IVAR, leg-level execution with realistic bid/ask crossing, DTE and
liquidity robustness, and clustered or bootstrap inference over a longer
history.
