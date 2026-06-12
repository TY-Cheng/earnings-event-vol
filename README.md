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

The current strategy/PnL layer uses `day_c2c` only. `jump_c2o` remains a primary
scientific decomposition target, but it is not reported as executable option
PnL in the current no-NBBO proxy run. The default tuning selection target is now
`day_c2c`, so validation-only hyperparameter selection is aligned with the
headline proxy economics. Trading decisions are evaluated in premium space. A
raw variance forecast is not enough; expected strategy value must beat market
entry cost and transaction cost estimates.

## Current State

Verified local state on 2026-06-13:

- `just data` now targets the main rebuild window `2016-10-01` through
  `2026-06-05`; pass explicit `args` only for bounded cache/debug reruns.
- `just research-fast` remains the quick no-sequence smoke-refresh command. The
  local pre-window-change data/feature artifact set was refreshed for the
  broader 2016-01-01 window, while the main `2016-10-01` window should be
  rebuilt on the remote data device before citing current results.
- `just research` builds the current proxy feature/model/report package
  from the current trade-proxy event panel, including sequence diagnostics.
- The populated local gold feature matrix is useful as a data/feature
  preflight snapshot; model/report metric artifacts should still be treated as
  stale until the model stage is rerun on the refreshed main-window feature
  matrix.
- The target rebuild and paper window is `2016-10-01` through `2026-06-05`, but
  paper-grade claims still require complete target-window quote/NBBO-equivalent
  coverage.
- All current trade-price results are `panel_grade=no_nbbo_trade_proxy` and
  `paper_grade=false`.

Latest local pre-window-change data rebuild artifacts:

- Options day-aggregate bronze coverage reaches the broader preflight window.
  The main `2016-10-01` window avoids the known 2016-H1 underlying daily
  entitlement gap.
- Dynamic SEC calendar: 5,785 universe-filtered candidate rows and 3,072
  BMO/AMC main-sample candidate events after SEC primary-document validation
  and SEC acceptance-time proxy timing.
- SEC CompanyFacts: 228,205 standardized fact rows for 201 tickers.
- Event-window panel: 3,072 events, 3,001 with realized event variance, 3,071
  with entry-window support, 80,275 contract candidates, 40,709 quote-pool
  contracts, 17,595 main DTE 5-14 contracts, and 39,566 IVAR-support-only
  contracts.
- Contract-reference validation: 79,903 unique option tickers, 79,634
  validated, 269 `missing_reference`, and 80,006 proxy-usable candidate rows.
  Unknown adjusted-deliverable state is excluded from proxy usability.
- Trade-proxy panel: 3,072 events, 3,001 with RVAR, 2,538 with trade-proxy
  IVAR, 80,006 proxy-usable contract rows, 55,580 contracts with usable
  pre-entry trade proxy marks, and 24,426 with no trade in the cutoff window.
- Second-aggregate cache: 79,920 main files with 70,825 writes and 9,095 hits;
  exit-preclose and post-open caches each have 5,400 files with 4,134 writes
  and 1,266 hits.
- Quote execution artifacts: a bounded targeted REST slice is populated with
  `--quote-workers 8`, cache reuse, and sixteen merged follow-on shards
  (`offset64_size64`, `offset128_size64`, `offset192_size64`,
  `offset256_size64`, `offset320_size16`, `offset336_size16`,
  `offset352_size16`, `offset368_size16`, `offset384_size16`,
  `offset400_size16`, `offset416_size16`, `offset432_size16`,
  `offset448_size16`, `offset464_size16`, `offset480_size16`,
  `offset496_size16`). It covers 502 events with 14,366 quote-window requests,
  10,921,438 matched quote rows, 14,366 window marks, 14,366 leg execution
  rows, 3,599 straddle rows, 502 quote-IVAR diagnostic rows, 7,183 quote-IV
  leg rows, 3,599 quote-IV surface-pair rows, 502 quote-surface IVAR rows, and
  502 confidence rows. The bounded surface has 7,164 finite `quote_mid_iv`
  values, 3,573 finite quote mid-total-variance rows, and 471 finite
  surface-IVAR mid rows. Confidence bands are 448 high, 53 medium, and 1 low.
  No full-day quote files are stored in the repo; full-sample quote/NBBO
  evidence is still pending. Follow-on quote runs can use `--quote-event-offset N`,
  `--max-events M`, and `--quote-batch-label offsetN_sizeM` to write
  batch-specific lake/artifact paths without overwriting the canonical bounded
  slice, then `--stage quote-execution-merge --quote-merge-batch offsetN_sizeM`
  to consolidate verified shards into canonical quote diagnostics.
- Lake quality audit: `lake-quality-audit` writes target-window coverage
  gates. For the main `2016-10-01` window, the expected remaining blocker is
  full bid/ask/NBBO-equivalent quote coverage rather than 2016-H1 underlying
  daily bars.

Latest proxy modeling artifacts:

- Feature matrix: 3,071 rows and 559 columns under `fe_v2_sec_xbrl`. The
  refreshed `feature_schema_report.csv` has 415 `model_feature=true` columns,
  including sequence call/put volume imbalance aggregates, own-underlying
  pre-event return/RV run-up, and SEC SIC coarse controls.
- Models evaluated: market-implied IVAR, last-four RVAR, last-four IVAR,
  Goyal-Saretto-style RV-IV spread, Elastic Net, LightGBM, XGBoost, a
  LightGBM/XGBoost forecast ensemble, FT-Transformer, and the refreshed
  sequence diagnostic suite.
- The Goyal-Saretto-style row is an earnings-event RV-IV spread benchmark
  inspired by the original predictability literature, not a full
  cross-sectional option-portfolio replication.
- Current tuned protocol: train/validation-locked tuning parameters are selected
  before locked-test evaluation, and `sequence_suite=all` is diagnostic unless
  common-row/control/bootstrap gates pass. Current code defaults to the
  `tuned_phase1_day_c2c_rank_log_rvar` profile: learned tabular models and
  FT-Transformer train on `log(max(RVAR, 0) + 1e-6)`, forecasts are
  back-transformed to variance units before metrics and strategy logic, and
  validation selection targets `day_c2c` edge ranking rather than direct PnL.
  It will not reuse stale `jump_c2o`, raw-target, or old-profile
  selected-parameter artifacts. The current Mac checkout materialized a
  broader pre-window-change data and feature matrix, but `lightgbm_tuned`
  segfaulted in the local LightGBM runtime during a no-sequence model smoke
  run; rebuild the 2016-10-01 main-window data/features and rerun models/report
  on another stable CPU/GPU environment before citing current selected params,
  model rows, or PnL.
- The active sequence diagnostic suite is ridge-flat, attention pooling,
  dilated CNN, mask-only, and time-shuffle. It uses only lightweight in-repo
  PyTorch encoders; the slow recurrent/SSM 5-seed sequence ensembles are not
  active model ids or runtime dependencies. Sequence rows remain diagnostic
  unless the common-row/control/bootstrap/economics gates pass. After the
  sequence-control runtime cleanup, rerun models/report before citing
  mask-only or time-shuffle numeric rows as current-code evidence.
- The feature stage emits the canonical hybrid sequence tensor only at
  `$GOLD_DATA_DIR/modeling/hybrid_sequence_tensor_v2.npz`; the non-`_v2`
  compatibility duplicate is no longer written.
- `FT-Transformer` refers to the validation-tuned tabular transformer
  specification.
- In the refreshed `fe_v2_sec_xbrl` sequence-suite snapshot, the strongest `jump_c2o` AUC
  is the Goyal-Saretto-style spread at about 0.620, while the old
  LightGBM/XGBoost ensemble row has the best `jump_c2o` OOS R2 versus IVAR at
  about 0.236. Rerun models under
  `tuned_phase1_day_c2c_rank_log_rvar` before citing any new selected params or
  the dual-output LightGBM/XGBoost ensemble row.
- The refreshed `day_c2c` headline proxy economics remain negative across
  tabular rows; the best net proxy PnL is still Goyal-Saretto-style spread at
  about -1,948 USD. This weakens any direct executable-trading sell and
  supports a more conservative signal-screening/market-efficiency framing.
- `reaction_o2c` is included in the proxy model artifacts as a diagnostic
  target. In the current sequence refresh, ridge-flat sequence leads O2C AUC at
  about 0.808, but O2C strategy rows remain `pnl_headline_eligible=false`.
- The sequence gate does not upgrade the claim: for primary `jump_c2o`,
  sequence rows fail the control/bootstrap gate.

## Command Surface

Use `just` as the public command surface:

```bash
just status
just check
just data args="--dry-run"
just data
just research
just research-fast
just research-report
just docs
```

`just status` is a lightweight environment diagnostic. It checks the resolved
repo, `DATA_DIR`, report/artifact roots, and configured secret-file paths
without downloading data, rebuilding artifacts, or training models.

`just check` is the full handoff gate. It formats, fixes lint, runs mypy,
pytest with the 95% coverage threshold, MkDocs strict build, docs-figure sync
checks, `status`, and source probes.

`just data` runs the active `all` data DAG:

```text
options-day-aggs-bulk -> universe -> dynamic-calendar -> sec-companyfacts
  -> event-window-panel -> contract-reference-validation -> trade-proxy-panel
  -> quote-execution-panel
```

Default data parameters:

- target rebuild range: `2016-10-01` to `2026-06-05`;
- bounded debug rerun ranges should be passed explicitly when needed;
- universe lookback for the target rebuild: from `2016-04-01`;
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
  - targeted Massive REST quote windows for quote execution diagnostics. The
    pipeline does not store full-day raw quote files: bronze
    stores quote-window requests and matched normalized quote subsets, silver
    stores selected quote marks plus leg-level bid/ask execution diagnostics,
    and gold stores straddle execution diagnostics plus event-level execution
    confidence.
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
- `quote-execution-panel` defaults to metadata-only planning in the active
  data DAG. Use `--stage quote-execution-panel --quote-run --quote-date
  YYYY-MM-DD` for a bounded quote scan; full-date-range quote streaming
  requires the explicit `--quote-allow-all-dates` guard. For resumable slices,
  add `--quote-event-offset N --max-events M --quote-batch-label offsetN_sizeM`
  so the run writes under `batches/batch=...` and leaves canonical outputs
  untouched. After verifying one or more batch shards, run
  `--stage quote-execution-merge --quote-merge-batch offsetN_sizeM` to update
  the canonical quote lake and research-facing CSV/report artifacts.
- SEC CompanyFacts is public XBRL financial-statement data. The active stage
  uses CIK-mapped CompanyFacts with conservative as-of gating:
  `acceptanceDateTime <= feature_asof_timestamp` when available, otherwise
  `filed < feature_asof_date`.

`just research` does not download market data. For the paper-facing snapshot,
run the canonical tuned proxy package with the full sequence diagnostic suite
and 1,000 bootstrap iterations:

```bash
just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1_day_c2c_rank_log_rvar --feature-schema-version fe_v2_sec_xbrl"
```

In the canonical `tuned_phase1_day_c2c_rank_log_rvar` protocol, Optuna
objectives and `ElasticNetCV` read only train and locked validation rows.
Elastic Net, LightGBM, XGBoost, and FT-Transformer train in log-RVAR space
with `FORECAST_FLOOR=1e-6`, then back-transform forecasts to raw variance
units before forecast metrics, ranking, strategy, IVAR-defeat, and casebook
artifacts. The default selection target is `day_c2c`; selected hyperparameters
are refit on train+validation, and locked test rows are evaluated once after
selection. Proxy PnL remains economic validation, not the hyperparameter
objective. `jump_c2o` remains a scientific target tested with the same
selected hyperparameters for cross-target generalization.
Paired original rows are intentionally not emitted.

The default feature schema is `fe_v2_sec_xbrl`. It uses the resolved
`artifacts/modeling/feature_schema_report.csv` as the model-feature allowlist,
excludes raw IDs and outcome/exit/PnL fields, adds point-in-time rolling
same-ticker earnings history, SEC XBRL fundamentals, SEC SIC coarse controls,
train-fitted rank/z-score features, single-name run-up/surface proxy features,
call/put volume imbalance, and own-underlying pre-event return/RV run-up.

It consumes the current proxy panel, builds features, trains/evaluates models,
writes metrics, writes `reports/modeling/proxy_research_report.md`, regenerates
`reports/modeling/figures/*.png`, and syncs those figures into
`docs/assets/images/modeling/`.

`just research-report` regenerates only the generated report and figure assets
from existing modeling artifacts. The curated reader-facing
`docs/results_snapshot.md` is intentionally manual: update it when a run changes
the paper-facing tables or interpretation, then run `just check`.

## Key Outputs

Data pipeline:

- `artifacts/data_pipeline/data_pipeline_manifest.json`
- `artifacts/data_pipeline/universe/universe_manifest.json`
- `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json`
- `artifacts/data_pipeline/sec_companyfacts/sec_companyfacts_manifest.json`
- `artifacts/data_pipeline/sec_companyfacts/sec_companyfacts_diagnostics.csv`
- `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json`
- `$GOLD_DATA_DIR/event_panel/trade_proxy_event_panel.parquet`

Research package:

- `$GOLD_DATA_DIR/modeling/feature_matrix.parquet`
- `artifacts/modeling/feature_schema_report.csv`
- `artifacts/modeling/feature_transform_params.json`
- `artifacts/modeling/forecast_metrics.csv`
- `artifacts/modeling/ranking_metrics.csv`
- `artifacts/modeling/strategy_metrics.csv`
- `artifacts/modeling/strategy_breakdowns.csv`
- `artifacts/modeling/ivar_defeat_breakdowns.csv`
- `artifacts/modeling/robustness_summary.csv`
- `artifacts/modeling/model_fit_diagnostics.csv`
- `artifacts/modeling/model_predictions.parquet`
- `artifacts/modeling/sequence_v2_quality.csv`
- `artifacts/modeling/common_row_pairwise_metrics.csv`
- `artifacts/modeling/incremental_value_diagnostics.csv`
- `artifacts/modeling/sequence_model_fit_diagnostics.csv`
- `artifacts/modeling/completion_gap_audit.csv`
- `artifacts/modeling/completion_gap_audit.json`
- `artifacts/data_pipeline/lake_quality_audit/lake_dataset_coverage.csv`
- `artifacts/data_pipeline/lake_quality_audit/lake_year_coverage.csv`
- `artifacts/data_pipeline/lake_quality_audit/lake_quality_report.json`
- `reports/modeling/proxy_research_report.md`
- `reports/modeling/figures/`

## Claim Boundaries

Current evidence supports engineering and signal-screening discussion only.
It does not support final paper claims that require bid/ask or NBBO execution.

Do not claim:

- generic IV forecasting superiority;
- paper-grade full-spread tradability;
- that second-aggregate trade bars are NBBO quotes;
- that a sequence architecture is the contribution independent of baselines and costs;
- that lower RMSE alone implies economic value.

The defensible near-term claim is narrower:

> In a no-NBBO proxy sample, state and event-history features show preliminary
> cross-sectional ranking signal for earnings event-variance mispricing beyond
> the market-implied IVAR baseline. Paper-grade claims require quote/NBBO data,
> robust cost/inference checks, and a full 2016-10-01 to 2026-06-05 rebuild.

## Docs

- Home: project object and current status.
- Results Snapshot: current artifacts and readiness boundaries.
- Paper Plan: research design and model/backtest protocol.
- Audit Prompts: implementation and manuscript review checklists.
- Future Work: paper blockers and deferred extensions.

`SPEC.md` is the implementation and research-protocol contract. It stays at the
repo root and is not a separate docs-nav page.
<!-- --8<-- [end:docs-home] -->
