---
hide:
  - navigation
---

# Results and Discussion

This page is the paper-facing results ledger for the current cold-run proxy
research package. It should be read with [Paper Plan](paper_plan.md).

Current canonical evidence root:

```text
/home/tycheng/data/earnings-event-vol
```

Repo-local `artifacts/` and `reports/` are ignored local outputs and are not
the current canonical evidence source. Selected figures from the cold-run
report are copied into `docs/assets/images/modeling/`.

## 3. Results and Discussion

### 3.1 Execution Grade and Reproducibility Status

The current run is a complete proxy-stage research package for
2016-10-01 through 2026-06-05. It is not paper-grade executable trading
evidence.

| Item | Current value |
| --- | --- |
| Evidence grade | `no_nbbo_trade_proxy` |
| Data root | `/home/tycheng/data/earnings-event-vol` |
| Target window | 2016-10-01 to 2026-06-05 |
| Feature schema | `fe_v2_sec_xbrl` |
| Tuning profile | `tuned_phase1_day_c2c_rank_log_rvar` |
| Split | chronological 70/15/15 |
| Forecast floor | `1e-6` |
| Model stage | `ok=true` |
| Report stage | `ok=true` |
| Report figures | 11 |

Lake-quality audit:

| Item | Value |
| --- | ---: |
| Audit `ok` | `false` |
| Required datasets | 15 |
| Incomplete required datasets | 13 |
| Paper-grade execution ready | `false` |

The blocking audit message is:

```text
requires full-window quote/NBBO or equivalent coverage and quote-IVAR beyond
the current bounded diagnostic slice
```

**Discussion.** The current package is reproducible and analysis-complete for
proxy research. It is not a final paper-grade execution package because the
lake-quality and completion-gap audits explicitly block full bid/ask or
NBBO-equivalent claims.

### 3.2 Research Question and Evidence Map

The paper-facing question is:

> Can models improve trading decisions around option-implied earnings event
> variance mispricing?

The evidence hierarchy is:

| Evidence layer | Current status |
| --- | --- |
| Forecast fit | Populated for all 3 targets and 9 active model ids. |
| Ranking | Populated for all 3 targets and 9 active model ids. |
| Strategy proxy | Populated, but positive rows are small-trade and proxy-only. |
| Quote confidence | Populated across train/validation/test and all targets. |
| IVAR defeat | Populated at event, metric, and breakdown levels. |
| Casebook | Populated for false positives, false negatives, market-vs-model cases, and quote-confidence splits. |
| Paper-grade execution | Blocked. |

### 3.3 Sample Construction and Coverage

| Artifact | Current value |
| --- | ---: |
| Feature matrix rows | 2,388 |
| Feature-schema rows | 569 |
| Model features | 407 |
| Event-level model features | 249 |
| Tree model features | 407 |
| Prediction rows | 7,164 |

Quote-execution diagnostics:

| Artifact | Current value |
| --- | ---: |
| Quote execution route | `quote_batch_consolidation` |
| Quote-confidence events | 2,329 |
| Quote-window requests | 65,172 |
| Matched quote rows | 21,680,332 |
| Event windows without quote confidence | 60 |
| Targeted request events with zero returned quote rows | 923 |
| Full-day quote files written | `false` |

Completion-gap audit:

| Item | Current value |
| --- | --- |
| `ok` | `false` |
| `paper_grade_ready` | `false` |
| Status counts | `complete=8`, `diagnostic_only=1`, `incomplete=3` |
| Blocking requirements | sequence full-suite population, target-window data coverage, paper-grade bid/ask/NBBO execution, quote-IVAR/surface paper-grade upgrade |

**Discussion.** The quote route has moved from entitlement theory to concrete
targeted extraction and diagnostics. However, the audit still blocks the
upgrade from targeted diagnostic quote evidence to full-window paper-grade
execution evidence.

The source coverage audit shows the previous missing-event concern was not an
early-window-only source claim. The verified merge covers contiguous target
offsets 0-2388, and the remaining 60 event windows without quote confidence
are classified as 22 no-candidate cases and 38 no-quote-eligible cases. The
923 `missing` quote-confidence cases had targeted REST requests but returned
zero quote rows; this is bounded evidence for those windows, not full
historical NBBO unavailability.

### 3.4 Model Suite

The current cold run evaluates 9 active model ids for each of 3 targets:

| Family | Models |
| --- | --- |
| Market and historical benchmarks | `market_implied_event_variance`, `last_four_rvar`, `last_four_ivar` |
| Classical spread benchmark | `goyal_saretto_rv_iv_spread` |
| Learned tabular/deep models | `linear_elastic_net_tuned`, `lightgbm_tuned`, `xgboost_tuned`, `lightgbm_xgboost_forecast_ensemble`, `ft_transformer` |
| Sequence tensor diagnostics | hybrid tensor v2, sequence coverage, quality reports, retired-model manifest |

**Discussion.** The model ladder is strong enough for a benchmark-disciplined
working paper. The verified refresh used `sequence_suite=none`; sequence
tensors and quality reports are present, but sequence model rows are not active
current metric evidence.

### 3.5 Forecast and Ranking Results

Best ranking AUC by target:

| Target | Best model | AUC | Top-decile precision |
| --- | --- | ---: | ---: |
| `day_c2c` | `lightgbm_xgboost_forecast_ensemble` | 0.5823 | 0.4483 |
| `jump_c2o` | `goyal_saretto_rv_iv_spread` | 0.5091 | 0.2069 |
| `reaction_o2c` | `ft_transformer` | 0.6714 | 0.2414 |

Best forecast OOS R2 versus IVAR by target:

| Target | Best model | OOS R2 vs IVAR | RMSE |
| --- | --- | ---: | ---: |
| `day_c2c` | `xgboost_tuned` | 0.6438 | 0.0231 |
| `jump_c2o` | `xgboost_tuned` | 0.7301 | 0.0208 |
| `reaction_o2c` | `linear_elastic_net_tuned` | 0.9312 | 0.0082 |

Primary `day_c2c` ranking top rows:

| Model | AUC | Top-decile precision | Edge-decile Spearman |
| --- | ---: | ---: | ---: |
| `lightgbm_xgboost_forecast_ensemble` | 0.5823 | 0.4483 | 0.5394 |
| `xgboost_tuned` | 0.5821 | 0.4483 | 0.4424 |
| `lightgbm_tuned` | 0.5818 | 0.4483 | 0.4303 |
| `linear_elastic_net_tuned` | 0.5742 | 0.4483 | 0.6000 |
| `ft_transformer` | 0.5718 | 0.4138 | 0.6970 |
| `goyal_saretto_rv_iv_spread` | 0.5503 | 0.2414 | 0.1636 |

Primary `day_c2c` forecast top rows:

| Model | OOS R2 vs IVAR | RMSE | MAE |
| --- | ---: | ---: | ---: |
| `xgboost_tuned` | 0.6438 | 0.0231 | 0.0109 |
| `lightgbm_xgboost_forecast_ensemble` | 0.6425 | 0.0231 | 0.0109 |
| `lightgbm_tuned` | 0.6408 | 0.0232 | 0.0109 |
| `linear_elastic_net_tuned` | 0.6307 | 0.0235 | 0.0111 |
| `last_four_rvar` | 0.5604 | 0.0257 | 0.0147 |

**Discussion.** Forecast-fit evidence is stronger than economic evidence.
XGBoost and the LightGBM/XGBoost ensemble are close on `day_c2c` ranking and
forecast fit. This does not establish sequence superiority because sequence
model rows were not populated in the verified refresh.

### 3.6 Headline `day_c2c` Proxy Strategy Results

Best `day_c2c` headline strategy rows by net PnL:

| Model | Strategy proxy | Trades | Net PnL | Return on premium | Sharpe | Hit rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `xgboost_tuned` | C2C exit preclose 15m | 3 | 1,484.41 | 0.4169 | 0.9498 | 0.6667 |
| `lightgbm_tuned` | C2C exit preclose 15m | 1 | 78.57 | 0.1448 | n/a | 1.0000 |
| `lightgbm_xgboost_forecast_ensemble` | C2C exit preclose 15m | 2 | -45.69 | -0.0525 | -0.2253 | 0.5000 |
| `linear_elastic_net_tuned` | C2C exit preclose 15m | 3 | -125.66 | -0.0954 | -0.6883 | 0.3333 |
| `goyal_saretto_rv_iv_spread` | C2C exit preclose 15m | 5 | -280.19 | -0.0198 | -0.0695 | 0.6000 |
| `last_four_rvar` | C2C exit preclose 15m | 5 | -792.10 | -0.0625 | -0.2093 | 0.6000 |

**Discussion.** This is not a robust positive PnL result. The best net row is
XGBoost with only 3 trades, and the second positive learned-model row has only
1 trade. The paper should sell ranking/diagnostic value, not economic
outperformance.

### 3.7 `jump_c2o` Diagnostics

Best `jump_c2o` ranking AUC is the Goyal-Saretto-style spread at 0.5091.
Best forecast OOS R2 versus IVAR is XGBoost at 0.7301. The best proxy strategy
row is the Goyal-Saretto-style spread on the 5-15 minute post-open option-VWAP
diagnostic with 5 trades and 3,089.51 USD net PnL.

**Discussion.** `jump_c2o` is scientifically important because it isolates the
overnight earnings jump. The strategy rows remain diagnostic because the
execution path is not a full paper-grade bid/ask/NBBO route and trade counts
are very small.

### 3.8 `reaction_o2c` Diagnostics

Best `reaction_o2c` ranking AUC is FT-Transformer at 0.6714. Best forecast
OOS R2 versus IVAR is Elastic Net at 0.9312. The best proxy strategy row is the
Goyal-Saretto-style spread with 4 trades and 1,637.39 USD net PnL.

**Discussion.** O2C validates that post-open digestion contains a different
signal shape, but it should not be sold as the main event-variance trading
claim.

### 3.9 Feature Schema Hygiene

The current feature schema has 569 report rows and 407 model features. The
feature allowlist excludes raw identifiers, post-event outcomes, PnL fields,
quote execution confidence, quote-IVAR, quote-IV surface diagnostics, and
unsupported NBBO claims.

**Discussion.** The feature schema is now strong enough for reproducible
proxy-stage modeling. Future feature work should start with ablation and
pruning, not broad feature expansion.

### 3.10 Quote-Aware Diagnostics

Quote-confidence prediction coverage has 36 rows across train, validation,
test, targets, and confidence bands.

Quote-IVAR summary:

| Execution confidence band | Events | Mid-IVAR available | Ask-IVAR available | Median confidence score | Median spread/mid |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 1,035 | 989 | 617 | 0.9444 | 0.0745 |
| medium | 334 | 217 | 1 | 0.7083 | 0.2206 |
| low | 37 | 0 | 0 | 0.3542 | 0.4500 |
| missing | 982 | 0 | 0 | 0.0000 | n/a |

**Discussion.** Quote confidence is useful for explaining which cases have
cleaner execution diagnostics. It does not by itself create paper-grade
tradability.

### 3.11 IVAR Defeat and Casebook

Current artifacts:

| Artifact | Shape / rows |
| --- | ---: |
| `ivar_defeat_events.csv` | 7,641 rows |
| `ivar_defeat_metrics.csv` | 27 rows |
| `ivar_defeat_breakdowns.csv` | 3,078 rows |
| `casebook_events.csv` | 2,722 rows |
| `casebook_summary.csv` | 126 rows |
| `quote_confidence_casebook_summary.csv` | 317 rows |

**Discussion.** These artifacts make the failure analysis paper-ready in
structure: false positives, false negatives, model-corrects-market,
market-right-model-wrong, and execution-fragile cases can now be discussed
systematically. The interpretation must still respect the proxy-stage
execution boundary.

### 3.12 Calibration, QLIKE, Robustness, and Sequence Gate

The current report includes calibration, QLIKE, cost sensitivity, robustness,
common-row pairwise metrics, sequence tensor diagnostics, and incremental-value
diagnostics.

Completion-gap blockers include `sequence_diagnostics_full_suite_populated`,
so sequence model claims should remain deferred until a deliberate
sequence-suite run is completed.

**Discussion.** The metric hierarchy in the paper plan is supported: ranking
and tail selection are more relevant to the research question than raw RMSE
alone, but strategy evidence remains too fragile for a trading-performance
claim.

### 3.13 Current Sellable Claim

The defensible claim is:

> We build a reproducible earnings event-variance mispricing research pipeline
> with strong benchmark discipline, log-target model tuning, targeted
> quote-aware diagnostics, IVAR-defeat analysis, and casebook interpretation.
> The current evidence supports signal screening, not paper-grade executable
> option-trading outperformance.

The current package should not claim:

- paper-grade executable trading performance;
- full bid/ask, OPRA, or NBBO execution;
- sequence-model superiority;
- robust positive strategy PnL;
- complete full-window quote-IVAR/surface coverage.

### 3.14 Next Evidence Required

| Requirement | Current status |
| --- | --- |
| Full-window quote/NBBO or equivalent coverage | Blocked by lake audit. |
| Full-sample quote-IVAR/surface evidence | Diagnostic only; completion gap remains. |
| Paper-grade bid/ask crossing | Not complete. |
| Sequence full-suite model rows | Not populated in the verified refresh. |
| Robust economic claim | Not supported by current small-trade positive rows. |

The next paper decision is whether to present this as a conservative
signal-screening paper or invest in full-window quote/NBBO-equivalent coverage
before submission.
