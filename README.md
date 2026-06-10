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

Verified local state on 2026-06-11:

- `just data` builds the active no-NBBO proxy data pipeline with a 2013-2025
  target window; pass `args="--start 2022-12-01 --end 2025-12-31"` for a
  bounded current-cache rerun.
- `just research-fast` remains the quick no-sequence smoke-refresh command. The
  current verified artifact set was refreshed with `sequence_suite=all`, reusing
  locked tuning parameters and keeping sequence rows diagnostic.
- `just research` still builds the full V5 proxy feature/model/report package
  from the current trade-proxy event panel, including sequence diagnostics.
- `just mamba-install` installs the local CUDA Mamba wheels and `just
  mamba-doctor` verifies the official `mamba-ssm` runtime.
- Current data range is `2022-12-01` through `2025-12-31`, because the observed
  Massive options day-aggregate entitlement in this workspace starts in 2022.
- The target paper range remains 2013-2025, but that needs upgraded historical
  option data entitlement or another licensed options route.
- All current trade-price results are `panel_grade=no_nbbo_trade_proxy` and
  `paper_grade=false`.

Latest proxy data artifacts:

- Dynamic calendar/event window panel: 816 BMO/AMC main-sample events after the
  current restored SEC-first calendar and universe filters.
- Trade-proxy panel: 816 events, 807 with the backward-compatible C2C
  `rvar_event` alias, 705 with trade-proxy `IVAR_event`.
- Event contract candidates: 23,845 total; 11,729 quote-pool contracts; 11,729
  missing-reference standard-contract fallback rows allowed by the current
  contract-reference policy.
- Proxy entry-price status: 10,046 contracts with usable pre-cutoff
  second-aggregate prices and 1,683 with no trade in the cutoff window.
- Proxy straddle diagnostics: 789 rows; mean gross C2C primary exit-preclose
  VWAP proxy PnL about -112.06 USD, mean exit-preclose VWAP proxy PnL about
  -162.30 USD, and mean haircut proxy PnL about -289.80 USD.
- Quote execution artifacts: a bounded targeted REST slice is populated with
  `--quote-workers 8` and cache reuse. It covers 64 events with 1,642
  quote-window requests, 1,226,559 matched quote rows, 1,642 window marks, 1,642
  leg execution rows, 412 straddle rows, 64 quote-IVAR diagnostic rows, 821
  quote-IV leg rows, 412 quote-IV surface-pair rows, 64 quote-surface IVAR
  rows, and 64 confidence rows. The bounded surface has 821 finite
  `quote_mid_iv` values, 412 finite quote total-variance rows, and 57 finite
  surface-IVAR mid rows. Confidence bands are 55 high and 9 medium. No full-day
  quote files are stored in the repo; full-sample quote/NBBO evidence is still
  pending. Follow-on quote runs can use `--quote-event-offset N --max-events M
  --quote-batch-label offsetN_sizeM` to write batch-specific lake/artifact paths
  without overwriting the canonical 64-event slice, then
  `--stage quote-execution-merge --quote-merge-batch offsetN_sizeM` to
  consolidate verified shards into canonical quote diagnostics.
- Lake quality audit: `lake-quality-audit` now writes 2013-2025 coverage gates.
  The latest audit finds 17/17 audited lake datasets are span-incomplete for
  the target window, including all 15 required paper-grade datasets. Options
  day aggregates cover 2022-05-04 to 2025-12-31; underlying day aggregates cover
  2016-05-04 to 2025-12-31; the main event/modeling sample still starts in
  December 2022.

Latest proxy modeling artifacts:

- Feature matrix: 816 rows.
- Models evaluated: market-implied IVAR, last-four RVAR, last-four IVAR,
  Goyal-Saretto-style RV-IV spread, Elastic Net, LightGBM, XGBoost, a
  LightGBM/XGBoost rank-average ensemble, FT-Transformer, and the refreshed
  sequence diagnostic suite.
- Current fast tuned protocol: train/validation-locked tuning parameters are
  reused, locked test rows are evaluated once, `sequence_suite=all`, and the
  latest local model manifest uses `bootstrap_iter=200`. This refresh produced
  2,448 prediction rows, 48 forecast metric rows, 48 ranking metric rows, 96
  strategy metric rows, 4,830 IVAR defeat event rows, and 4,271 casebook event
  rows. Locked-test predictions now include 24 high/medium quote-confidence
  target rows across 8 unique events. The model manifest also writes a
  21-row sequence diagnostic gate table, a 21-row incremental-value table, an
  18-row robustness summary for DTE, liquidity, VIX-regime, timing, ticker, and
  quote-confidence splits, plus quote-confidence summary artifacts for
  prediction coverage, quote-IVAR, strategy, IVAR defeat, and casebook
  diagnostics.
- The sequence diagnostic suite was refreshed with explicit 5-seed sequence
  ensembles. Ridge-flat, BiGRU 5-seed, official `mamba-ssm` 5-seed, attention
  pooling, dilated CNN, mask-only, and time-shuffle rows all trained in the
  external CUDA-enabled uv environment. Sequence rows remain diagnostic unless
  the common-row/control/bootstrap/economics gates pass.
- `FT-Transformer` refers to the validation-tuned tabular transformer
  specification.
- In the refreshed FE V2 fast run, the strongest `jump_c2o` AUC is the
  Goyal-Saretto-style spread at about 0.620, while the best `jump_c2o` OOS R2
  versus IVAR is the LightGBM/XGBoost ensemble at about 0.236.
- The refreshed `day_c2c` headline proxy economics remain negative across
  tabular rows; the best net proxy PnL is still Goyal-Saretto-style spread at
  about -1,948 USD. This weakens any direct executable-trading sell and
  supports a more conservative signal-screening/market-efficiency framing.
- `reaction_o2c` is included in the V5 proxy model artifacts as a diagnostic
  target. In the current sequence refresh, ridge-flat sequence leads O2C AUC at
  about 0.808, but O2C strategy rows remain `pnl_headline_eligible=false`.
- The sequence gate does not upgrade the claim: for primary `jump_c2o`, all
  real sequence rows, including official `mamba-ssm` 5-seed, fail the
  control/bootstrap gate.

## Command Surface

Use `just` as the public command surface:

```bash
just status
just check
just mamba-doctor
just mamba-install
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
just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v2_sec_xbrl"
```

In the canonical tuned protocol, Optuna objectives and `ElasticNetCV` read only train and
locked validation rows. The selected hyperparameters are refit on
train+validation, and locked test rows are evaluated once after selection.
Paired original rows are intentionally not emitted.

The default feature schema is `fe_v2_sec_xbrl`. It uses the resolved
`artifacts/modeling/feature_schema_report.csv` as the model-feature allowlist,
excludes raw IDs and outcome/exit/PnL fields, adds point-in-time rolling
same-ticker earnings history, SEC XBRL fundamentals, train-fitted rank/z-score
features, and single-name run-up/surface proxy features. `fe_v1_legacy` remains
available only for same-code feature-ablation reruns.

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
- that Mamba is the contribution independent of baselines and costs;
- that lower RMSE alone implies economic value.

The defensible near-term claim is narrower:

> In a no-NBBO proxy sample, state and event-history features show preliminary
> cross-sectional ranking signal for earnings event-variance mispricing beyond
> the market-implied IVAR baseline. The current same-code ablation says the
> parsimonious FE V1 tabular signal is stronger than the richer FE V2 default,
> so FE V2 is a negative diagnostic result rather than a headline improvement.
> Paper-grade claims require quote/NBBO data and robust cost/inference checks.

## Docs

- Home: project object and current status.
- Results Snapshot: current artifacts and readiness boundaries.
- Paper Plan: research design and model/backtest protocol.
- Audit Prompts: implementation and manuscript review checklists.
- Future Work: paper blockers and deferred extensions.

`SPEC.md` is the implementation and research-protocol contract. It stays at the
repo root and is not a separate docs-nav page.
<!-- --8<-- [end:docs-home] -->
