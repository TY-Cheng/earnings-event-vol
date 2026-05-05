<!-- --8<-- [start:docs-home] -->
# Earnings Event Vol

Reproducible research pipeline for U.S. equity-options earnings event variance
forecasting and risk-defined options backtests.

The project question is not whether a deep model can reduce generic implied-volatility
RMSE. The paper-facing question is:

> Can machine learning improve trading decisions around option-implied earnings
> event variance mispricing?

The empirical target is the event-level comparison between realized earnings move
variance and the market-implied event variance extracted from short-dated options.
The main economic test is whether a model improves the ranking of tradable events
after realistic bid-ask costs and risk-defined option payoffs.

## Research Frame

Working title:

**Can Deep Learning Improve Earnings Volatility Trading? Evidence from U.S. Equity Options**

Technical title:

**State-Selective Event Variance Forecasting for Earnings Options: A Mamba-Based Approach with Risk-Defined Backtests**

Core target:

```text
RVAR_event > IVAR_event
```

The prediction task is aligned with the trade, but the entry rule is evaluated
in premium space:

- predicted cheap event volatility: evaluate long ATM straddles;
- predicted rich event volatility: evaluate short iron flies;
- optional second-stage relative-value extension: vega-normalized calendar straddles.
- raw variance edge: `forecast_RVAR_event - IVAR_event`;
- trading edge: expected strategy value minus market entry cost in USD.

## Current Status

Current status: the repository has a working deterministic data-engineering
spine for event alignment, provisional IVAR extraction, and a Massive
second-aggregate trade-price proxy panel. It is not yet a paper-grade
quote/NBBO backtest or a model-training pipeline. The implemented market-data
route does not ingest historical option quote rows: it starts from Massive
option second aggregates for pre-cutoff entry prices and diagnostics, and uses
exit-date option day-aggregate closes for proxy exits.

Implemented now:

- `earnings-event-vol` project metadata.
- Repo-local `.env` / `UV_PROJECT_ENVIRONMENT` workflow.
- Credential-file-only Massive configuration.
- `earnings_event_vol` Python package and CLI for status, source checks, audits,
  event panels, and data-pipeline stages.
- V1 protocol implementation for event alignment, variance extraction, data
  audit, leakage audit, feature checks, model registry, and deterministic
  backtest smoke.
- Integrity guards for timezone-aware event timestamps, IVAR expiry coverage,
  explicit model implementation claims, and fail-closed audit outputs.
- SEC-first earnings-candidate builder with optional Massive 8-K text validation.
- Contract discovery that excludes non-standard OCC deliverables before option
  proxy-price pooling.
- Event-panel diagnostics for PCP-vs-spot forward source, ATM selection method,
  American option forward caveat, and possible preannouncement/prior-guidance
  review.
- Paper-facing docs front door and research plan.
- V1.5 Massive second-aggregate trade-proxy route, marked
  `no_nbbo_trade_proxy`, for screening event alignment, IVAR extraction, and
  proxy PnL without quote/NBBO data.

Not yet implemented:

- Paper-grade historical bid/ask or NBBO ingestion.
- Full top-50 proxy event panel construction with coverage/fallback diagnostics.
- Production model training for LightGBM, FT-Transformer, and Mamba.
- Paper-grade empirical results.

## Quick Start

This repo uses `uv` and `just`. The local virtual environment is controlled by
`.env`, which should remain machine-local:

```bash
UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/earnings-event-vol"
```

Run the default local gate:

```bash
just check
```

`just check` syncs the environment, runs `ruff format` and `ruff check --fix`,
then runs strict typing, tests, MkDocs strict build, status, and the Massive
credential probe. The test gate enforces at least 93% coverage on the active
package.

Public entrypoints:

```bash
just status
just audit
just data
just docs
```

`just audit` writes the required field audit outputs on tiny fixtures. `just
audit date=2025-02-05` switches the same entrypoint to a Massive S3 flat-file
sample gate for that trading date: it records metadata for day aggregates and
`quotes_v1`, downloads only the small day-aggregate files, and writes normalized
schema heads plus a readiness report. The `quotes_v1` file is probed for
availability/size only; it is not part of the current proxy pipeline.

`just data` is the resumable data-engineering entrypoint. By default it runs the
V1.5 Massive second-aggregate proxy route end to end:
`calendar-pilot -> pilot-panel -> trade-proxy-panel`. The default smoke scope is
`--start 2020-01-01`, `--end 2025-12-31`, `--max-events 10`, `--jobs 4`,
`--lookback-seconds 900`, `--second-agg-buffer-minutes 60`,
`--price-field option_vwap`, and DTE `3-21` so one contract-discovery pass can
support the main `5-14` sample and the robustness window. Existing outputs are
skipped unless `--force` is passed through `args`. Use `--dry-run` to print the
storage/API/exclusion estimate without writing data outputs. The explicit stage
names remain available for rebuilding or resuming individual steps. Long-running
stages print start/end progress, second-aggregate counts, and exit day-agg
download/cache statuses. Cached Parquet files are reused when readable with the
expected schema; corrupt second-agg or exit day-agg caches are repaired by
deleting and re-fetching the affected partition.

The dynamic-universe stage is available behind the same command surface:
`just data universe args="--options-day-aggs PATH"`. It builds
`ticker_month_liquidity.parquet` from normalized option day aggregates and then
monthly top-50 snapshots using trailing six-month option premium dollar volume
with a VWAP-first, close-fallback rule.

The market-data path is lake-first. Massive `.csv.gz` downloads are temporary
transfer files; they are converted immediately to compressed Parquet and then
removed. The working layout is:

- `data/bronze/`: source-preserving Massive tables partitioned by date,
  including cached option second aggregates used only for entry pricing and
  pre-cutoff feature/liquidity diagnostics. Post-cutoff second-aggregate bars
  are not retained.
- `data/silver/`: cleaned calendar, event-window, contract, and IVAR input
  tables.
- `data/gold/`: analysis-ready event panels and later feature/model inputs.
- `artifacts/`: manifests, readiness reports, and audit summaries.

Large table reads and writes use Polars + Parquet. Pandas remains acceptable
for tiny fixtures and small in-memory orchestration.

```bash
just data
just data args="--dry-run"
just data args="--force"
just data massive-probe args="--dates 2025-02-05 2025-02-06 --jobs 2"
just data universe args="--options-day-aggs data/bronze/massive/options_day_aggs"
just data calendar-pilot args="--force --start 2025-01-01 --end 2025-12-31"
just data pilot-panel args="--max-events 3 --force"
just data trade-proxy-panel args="--max-events 3 --jobs 2 --force"
just data contracts args="--events PATH --contracts PATH"
just data panel args="--events PATH --quotes PATH"
```

`just docs` also formats/lint-fixes first, then builds and serves the strict
docs site.

## Data Sources

Primary first-version source:

- Massive flat files and APIs for U.S. options and underlying equity data.
- SEC EDGAR company submissions for official historical 8-K Item 2.02 event
  candidates and acceptance timestamps.
- Massive 8-K text for parsed filing-text validation.

Research design data requirements:

- Current V1.5 trade-proxy panels use Massive option second aggregates as
  pre-cutoff trade-price OHLCV proxies. Exit diagnostics use the same option
  contracts' exit-date `options_day_aggs` close when available, with intrinsic
  payoff only as a flagged fallback for missing exit closes or 0DTE expiry.
  These outputs are explicitly marked `no_nbbo_trade_proxy` and are for signal
  screening, not full-spread executable strategy claims.
- Future paper-grade backtests require U.S. single-name option end-of-day quotes
  or NBBO-equivalent data with bid, ask, volume, open interest, strike,
  expiration, call/put flag, and underlying close. That quote/NBBO source is not
  implemented in the current data route.
- Contract metadata with multiplier, contract size, deliverable status, and
  corporate-action flags so non-standard OCC contracts can be excluded.
- Earnings event calendar with announcement date, accession/source id,
  source timestamp, text-validation status, and BMO/AMC/DMH/unknown timing.
- Market controls such as SPY returns, VIX, sector ETF returns, rates, dividends,
  and corporate-action filters when available.

Nasdaq earnings-calendar rows can be used as auxiliary expected-calendar
metadata, but they are not the primary historical timing source for this study.

The active candidate route is SEC-first:

```bash
PYTHONPATH=src uv run --env-file .env python -m earnings_event_vol.cli build-earnings-calendar \
  --tickers AAPL MSFT TSLA \
  --start 2026-01-01 \
  --end 2026-04-30 \
  --out artifacts/earnings_calendar_sample
```

This writes `earnings_calendar_candidates.csv` and `earnings_calendar_report.json`.
The output is an auditable candidate table, not yet the final paper panel.

First paper universe:

- Top 50 liquid U.S. single-name option stocks, 2013-2025, BMO/AMC only.
- Top-150 expansion is deferred until the top-50 proxy data lake and later
  paper-grade quote/IV route are stable.

Credential policy:

- Put secret values in files outside the repo.
- Point `MASSIVE_API_KEY_FILE` and `MASSIVE_FLAT_FILE_KEY_FILE` at those files.
- Set `SEC_USER_AGENT` to a research contact string before running SEC pulls.
- Do not store direct API keys in `.env`, docs, source, or tests.

## Claim Boundaries

- The main contribution is trading-decision improvement for earnings event
  variance mispricing, not generic IV forecasting.
- The market-implied event variance baseline is the benchmark to beat.
- Deep learning is useful only if it improves tradable top-decile ranking and net
  returns after realistic transaction costs.
- V1 trading thresholds compare USD expected strategy edge with USD transaction
  cost; the raw variance edge is never compared directly with option spreads.
- If LightGBM beats Mamba, the paper remains valid as evidence that event-level
  nonlinear tabular interactions are enough.
- If no model beats implied event variance after costs, the paper remains valid
  as evidence that earnings option markets are difficult to beat.

## Documentation

- [Home](https://ty-cheng.github.io/earnings-event-vol/): project object,
  current status, local workflow, data route, and claim boundaries.
- [Results Snapshot](https://ty-cheng.github.io/earnings-event-vol/results_snapshot/):
  current implementation state and readiness boundaries.
- [Paper Plan](https://ty-cheng.github.io/earnings-event-vol/paper_plan/):
  research design, related literature, target definition, features, model order,
  and backtest gates. The repo-level `SPEC.md` remains the implementation
  contract, but it is not a separate reader-facing navigation entry.
- [Future Work](https://ty-cheng.github.io/earnings-event-vol/future_work/):
  deferred extensions beyond the first paper.
- [Development Audit](https://ty-cheng.github.io/earnings-event-vol/development_audit_prompt/):
  implementation audit checklist.
- [Manuscript Audit](https://ty-cheng.github.io/earnings-event-vol/manuscript_audit_prompt/):
  paper-readiness checklist.
<!-- --8<-- [end:docs-home] -->
