# Earnings Event Vol

--8<-- [start:docs-home]

Empirical research pipeline for U.S. single-name equity-option earnings event
variance mispricing.

## Research Question

Can models improve trading decisions around option-implied earnings event
variance mispricing?

This is not generic implied-volatility forecasting. The object is the event
variance embedded in short-dated earnings options:

```text
ex post mispricing = RVAR_event - IVAR_event
```

Models forecast realized earnings-event variance, compare that forecast with
market-implied `IVAR_event`, and feed the predicted edge into a
premium-space, cost-aware strategy layer.

## Current Verified State

Last verified: 2026-06-13 on the WSL2/CUDA cold-run device.

Canonical cold-run data root:

```text
/home/tycheng/data/earnings-event-vol
```

The repo-local `artifacts/` and `reports/` directories are not canonical
current evidence. They are ignored local outputs and should either be deleted
or explicitly archived before handoff.

### Evidence Grade

Current evidence grade is still `no_nbbo_trade_proxy`.

The pipeline now has targeted quote-window diagnostics, but it is not a
paper-grade executable NBBO backtest.

Lake-quality audit for `2016-10-01` through `2026-06-05`:

| Item | Value |
| --- | ---: |
| Audit `ok` | `false` |
| Required datasets | 15 |
| Incomplete required datasets | 13 |
| Paper-grade execution ready | `false` |

Blocker:

```text
requires full-window quote/NBBO or equivalent coverage and quote-IVAR beyond
the current bounded diagnostic slice
```

### Cold-Run Data and Quote Artifacts

| Artifact | Verified value |
| --- | ---: |
| Target window | 2016-10-01 to 2026-06-05 |
| Quote execution route | `quote_batch_consolidation` |
| Quote-confidence events | 2,329 |
| Quote-window requests | 65,172 |
| Matched quote rows | 21,680,332 |
| Event windows without quote confidence | 60 |
| Targeted request events with zero returned quote rows | 923 |
| Full-day quote files written | `false` |
| Feature matrix rows | 2,388 |
| Feature-schema rows | 569 |
| Model features | 407 |
| Event-level model features | 249 |
| Tree model features | 407 |

Quote data are used as targeted event-window evidence. The pipeline stores
quote-window requests and matched normalized quote subsets, not full-day raw
quote files.

The source coverage audit currently supports a bounded statement only:
targeted REST quote-window requests returned no rows for 923 events, while 60
events never reached quote-confidence output because 22 had no contract
candidates and 38 had no quote-eligible contracts. This is not yet a
source-level proof of full historical NBBO unavailability.

### Cold-Run Research Artifacts

| Artifact | Verified value |
| --- | ---: |
| Research profile | `tuned_phase1_day_c2c_rank_log_rvar` |
| Forecast floor | `1e-6` |
| Split design | chronological 70/15/15 |
| Trained model-target evaluations | 27 |
| Prediction rows | 7,164 |
| Forecast metric rows | 27 |
| Ranking metric rows | 27 |
| Strategy metric rows | 54 |
| IVAR-defeat metric rows | 27 |
| Casebook summary rows | 126 |
| Quote-confidence prediction coverage rows | 36 |
| Report figures | 11 |

`research_manifest.json`, `research_report_manifest.json`, and
`completion_gap_audit.json` were generated under the external cold-run
artifact root.

Completion-gap audit:

| Item | Value |
| --- | --- |
| `ok` | `false` |
| `paper_grade_ready` | `false` |
| Status counts | `complete=8`, `diagnostic_only=1`, `incomplete=3` |
| Blocking requirements | sequence full-suite population, target-window data coverage, paper-grade bid/ask/NBBO execution, quote-IVAR/surface paper-grade upgrade |

## Targets

| Target | Role |
| --- | --- |
| `day_c2c` | Current proxy-PnL headline and canonical tuning target. |
| `jump_c2o` | Scientific close-to-open earnings jump target. |
| `reaction_o2c` | Post-open digestion diagnostic. |

The canonical tuning profile trains learned tabular models and FT-Transformer
on:

```text
log(max(RVAR, 0) + 1e-6)
```

Forecasts are back-transformed to variance units before metrics, ranking,
strategy, IVAR-defeat, and casebook outputs.

## Models

Benchmarks:

- `market_implied_event_variance`
- `last_four_rvar`
- `last_four_ivar`
- `goyal_saretto_rv_iv_spread`

Learned tabular/deep models:

- `linear_elastic_net_tuned`
- `lightgbm_tuned`
- `xgboost_tuned`
- `lightgbm_xgboost_forecast_ensemble`
- `ft_transformer`

Sequence diagnostics:

- sequence tensors and quality audits are built;
- the current verified model refresh used `sequence_suite=none`;
- sequence model ids are therefore not active metric rows in this snapshot.

The Goyal-Saretto row is a Goyal-Saretto-inspired earnings-event RV-IV spread
benchmark, not a full replication of the original cross-sectional portfolio
study.

Slow 5-seed recurrent/SSM sequence ensembles are retired and not active public
model ids.

## Current Results Snapshot

These are current cold-run outputs, not the older repo-local snapshot.

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

Best headline proxy strategy row by target:

| Target | Best model | Strategy proxy | Trades | Net PnL | Return on premium |
| --- | --- | --- | ---: | ---: | ---: |
| `day_c2c` | `xgboost_tuned` | C2C exit preclose 15m | 3 | 1,484.41 | 0.4169 |
| `jump_c2o` | `goyal_saretto_rv_iv_spread` | post-open option VWAP 5-15 | 5 | 3,089.51 | 0.2044 |
| `reaction_o2c` | `goyal_saretto_rv_iv_spread` | O2C 5-15 to C2C exit | 4 | 1,637.39 | 0.2917 |

Interpretation: ranking and forecast evidence exist, but the positive strategy
rows are too small to support a strong trading-performance claim. The sellable
claim is conservative signal screening with strong benchmark discipline and
quote-aware diagnostics.

## Command Surface

Use a centralized virtual environment outside the repo:

```bash
export UV_PROJECT_ENVIRONMENT=/home/tycheng/.venvs/earnings-event-vol
export UV_CACHE_DIR=/tmp/uv-cache
```

Use a machine-local `.env` for secrets and default local paths. Do not commit
secrets or machine-specific data roots.

Common commands:

```bash
just status
just check
just data args="--stage lake-quality-audit --start 2016-10-01 --end 2026-06-05 --force"
just research args="--stage features --feature-schema-version fe_v2_sec_xbrl"
just research args="--stage models --feature-schema-version fe_v2_sec_xbrl --reuse-tuning-params"
just research args="--stage report --feature-schema-version fe_v2_sec_xbrl"
just _sync-doc-figures
```

Quote extraction is intentionally staged and resumable:

```bash
just data args="--stage quote-execution-panel --start 2016-10-01 --end 2026-06-05 --quote-run --quote-allow-all-dates --quote-source rest --quote-workers 8 --quote-event-offset N --max-events M --quote-batch-label LABEL"
just data args="--stage quote-execution-merge --quote-merge-exclude-canonical --force"
```

## Claim Boundaries

Allowed:

- Current proxy-stage signal-screening evidence.
- Quote-aware diagnostic coverage and confidence stratification.
- Model-vs-IVAR defeat analysis and casebook discussion.
- Clear statement that paper-grade execution is blocked.

Not allowed:

- Paper-grade executable trading performance.
- Full bid/ask, OPRA, or NBBO execution.
- Full-window quote-IVAR/surface coverage.
- Sequence-model superiority.
- Economic outperformance from small-trade positive rows.

--8<-- [end:docs-home]

## Docs

- [Paper Plan](docs/paper_plan.md): manuscript structure, research question,
  literature positioning, data, preprocessing, models, metrics, expected
  experiments, and claim boundaries.
- [Results and Discussion](docs/results_snapshot.md): current cold-run results,
  tables, figures, interpretations, and remaining blockers.
- [Future Work](docs/future_work.md): paper-grade blockers and deferred work.
