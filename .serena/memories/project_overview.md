# Project Overview

Name: `earnings-event-vol`

Authoritative project root:

```text
/home/tycheng/projects/earnings-event-vol/earnings-event-vol
```

Use this subfolder Serena project. Do not use the parent wrapper
`/home/tycheng/projects/earnings-event-vol/.serena`; it is a project-activation
confusion source and should be absent.

## Purpose

Reproducible empirical research pipeline for U.S. single-name equity-option
earnings event variance mispricing.

Core question:

> Can models improve trading decisions around option-implied earnings event
> variance mispricing?

This is not generic implied-volatility forecasting. Models forecast realized
earnings-event variance, compare it with market-implied `IVAR_event`, and
evaluate predicted mispricing in premium-space proxy strategies.

## Current Verified Evidence

Last synchronized: 2026-06-13 after the WSL2/CUDA cold run.

Canonical evidence root:

```text
/home/tycheng/data/earnings-event-vol
```

Target window:

```text
2016-10-01 through 2026-06-05
```

Current evidence grade:

- `no_nbbo_trade_proxy`
- `paper_grade_execution_ready=false`
- signal-screening and quote-aware diagnostics only

Lake-quality audit:

- `ok=false`
- required datasets: 15
- incomplete required datasets: 13
- blocker: full-window quote/NBBO-equivalent coverage and quote-IVAR beyond
  the current diagnostic route are not proven.

## Data and Quote State

Cold-run quote execution:

- route: `quote_batch_consolidation`
- quote-confidence events: 2,329
- quote-window requests: 65,172
- matched quote rows: 21,680,332
- event windows without quote confidence: 60
- targeted request events with zero returned quote rows: 923
- full-day quote files written: false

Cold-run feature/model inputs:

- feature matrix rows: 2,388
- feature-schema rows: 569
- model features: 407
- event-level model features: 249
- tree model features: 407

Quote data are targeted event-window diagnostics. The pipeline stores request
tables and matched normalized quote subsets, not full-day raw quote files.

## Targets

- `day_c2c`: canonical tuning target and current proxy-PnL headline.
- `jump_c2o`: scientific close-to-open earnings jump target.
- `reaction_o2c`: post-open digestion diagnostic.

Canonical profile:

- `tuned_phase1_day_c2c_rank_log_rvar`
- learned tabular models and FT-Transformer train on
  `log(max(RVAR, 0) + 1e-6)`
- forecasts are back-transformed to variance units before metrics, ranking,
  strategy, IVAR-defeat, and casebook logic.

## Implemented Models

Benchmarks:

- `market_implied_event_variance`
- `last_four_rvar`
- `last_four_ivar`
- `goyal_saretto_rv_iv_spread`

Learned models:

- `linear_elastic_net_tuned`
- `lightgbm_tuned`
- `xgboost_tuned`
- `lightgbm_xgboost_forecast_ensemble`
- `ft_transformer`

Sequence diagnostics:

- sequence tensors and quality reports are built.
- Current verified model refresh used `sequence_suite=none`.
- Sequence model ids are not active metric rows in this snapshot.

Slow 5-seed recurrent/SSM sequence ensembles are retired and not active public
model ids. Mamba/BiGRU references in code/tests should be retired-id manifests
or tests proving they are not active.

## Current Results

Research outputs:

- trained model-target evaluations: 27
- prediction rows: 7,164
- forecast metric rows: 27
- ranking metric rows: 27
- strategy metric rows: 54
- IVAR-defeat metric rows: 27
- casebook summary rows: 126
- quote-confidence prediction coverage rows: 36
- report figures: 11

Best ranking AUC:

- `day_c2c`: `lightgbm_xgboost_forecast_ensemble`, AUC 0.5823
- `jump_c2o`: `goyal_saretto_rv_iv_spread`, AUC 0.5091
- `reaction_o2c`: `ft_transformer`, AUC 0.6714

Best forecast OOS R2 versus IVAR:

- `day_c2c`: `xgboost_tuned`, 0.6438
- `jump_c2o`: `xgboost_tuned`, 0.7301
- `reaction_o2c`: `linear_elastic_net_tuned`, 0.9312

Headline strategy interpretation:

- Best `day_c2c` net row is `xgboost_tuned`, 3 trades, +1,484.41 USD.
- Positive rows are too small-trade to support an economic alpha claim.
- Current sell is benchmark-disciplined signal screening, not executable
  trading outperformance.

## Docs Contract

- `docs/paper_plan.md`: manuscript plan in paper order.
- `docs/results_snapshot.md`: current Results and Discussion ledger.
- `README.md`: front-door current state and commands.
- Keep docs synchronized only from verified artifacts.
