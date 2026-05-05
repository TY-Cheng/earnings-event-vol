# Results Snapshot

## Current State

This repository is an early-stage research pipeline for the earnings event
volatility project. The current implementation can build audited candidate
events, provisional event panels, and a second-aggregate trade-price proxy
panel. It does not yet produce paper-grade quote/NBBO backtest results. The
active market-data route has no historical option quote rows: proxy entry marks
come from Massive option second aggregates, and proxy exit marks come from
exit-date option day-aggregate closes when available.

Active and current:

- Project name: `earnings-event-vol`.
- Python package: `src/earnings_event_vol`.
- Local environment: `UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/earnings-event-vol"`.
- Credential policy: Massive keys are file-only through `.env`.
- Verification front door: `just check`, including `ruff format`, `ruff check
  --fix`, strict mypy, pytest, MkDocs strict, status, and Massive credential
  probes.
- Test coverage floor: 93% on the active package. Current local test gate:
  65 tests passed with 93.10% total coverage.
- Data-audit gate: `just audit` for fixtures; `just audit date=YYYY-MM-DD` for
  a narrow Massive flat-file sample gate.
- Data-engineering gate: `just data` for resumable stages. The current stages
  are `fixture-audit`, `massive-probe`, `options-day-aggs-bulk`, `universe`,
  `dynamic-calendar`, `calendar-pilot`, `contracts`, `panel`, `pilot-panel`,
  `trade-proxy-panel`, and `proxy-all`;
  outputs are skipped only when present and when their saved parameter signature
  matches the requested run.
- `just data` defaults to `proxy-all`, which runs
  `options-day-aggs-bulk -> universe -> dynamic-calendar -> pilot-panel ->
  trade-proxy-panel` over the final proxy study range `2013-2025`; the universe
  lookback begins at `2012-07-01`, with 4 workers by default, a 900-second
  pricing lookback, a 60-minute resolved-close pre-cutoff buffer, monthly top
  50 by trailing six-month option premium dollar volume, and DTE `3-21`
  contract discovery. Use `args="--max-events 10"` for a downstream smoke run;
  it does not shrink the universe/calendar build. `--force` rebuilds derived
  outputs while reusing valid bronze caches; `--refresh-bronze` explicitly
  re-fetches flat-file and second-aggregate bronze partitions. Explicit stage
  names are still available for rebuilding the calendar, pilot panel, fixture
  audit, and legacy fixture panel smoke paths. It prints stage-level progress
  plus bulk day-agg, second-agg, and exit day-agg count/status updates during
  long runs.
- Data lake layout: Massive flat-file downloads are temporary transfer files.
  They are converted immediately into compressed Parquet under `data/bronze/`;
  full-market option/underlying day aggregates are cached under
  `data/bronze/massive/options_day_aggs/` and
  `data/bronze/massive/underlying_day_aggs/`; second-aggregate trade-proxy bars
  are cached under
  `data/bronze/massive/options_second_aggs/` for entry/pre-cutoff diagnostics
  only; exit option prices come from exit-date `options_day_aggs` closes.
  Cached Parquet partitions are reused if readable with the expected schema;
  corrupt flat-file, second-agg, or exit day-agg caches are repaired by deleting
  and re-fetching the affected partition.
  Cleaned intermediate tables go to `data/silver/`; analysis-ready panels go to
  `data/gold/`. `artifacts/` is reserved for manifests, readiness reports, and
  audit summaries.
- Dynamic universe and calendar scaffolding now sit in the default proxy DAG.
  `universe` builds monthly ticker liquidity from normalized option day
  aggregates and top-50 trailing six-month option premium dollar-volume
  snapshots; `dynamic-calendar` queries SEC EDGAR submissions plus official
  SEC primary filing documents for the universe ticker union and filters events
  by latest prior universe membership, writing `universe_month`,
  `universe_rank`, `in_universe`, and `universe_filter_status`. Massive 8-K
  text is optional auxiliary fallback, not a required calendar dependency.
- First real Massive flat-file probe: `just audit date=2025-02-05` succeeded
  for option day aggregates, option quotes metadata, and underlying day
  aggregates. Outputs are in `artifacts/massive_flat_file_probe/`. The
  `options_quotes_v1` observation is metadata/readiness only; current proxy data
  engineering does not ingest that quote file.
- Current data-readiness finding: `day_aggs_v1` supports contract parsing,
  underlying close sampling, and a provisional local-IV route from option close
  prices. It lacks bid/ask, open interest, quote condition, IV, and Greeks, so
  it cannot support paper-grade IVAR extraction or transaction-cost backtests by
  itself. The active near-term route therefore remains `no_nbbo_trade_proxy`.
  A later paper-grade route would need `options_quotes_v1` or another
  historical bid/ask/NBBO source plus IV/Greeks/OI or local IV from mids.
- First earnings-source probe: `artifacts/earnings_calendar_source_probe/`
  compares Nasdaq calendar rows, SEC EDGAR company submissions, SEC primary
  filing documents, and Massive 8-K text for AAPL, AMZN, MSFT, NVDA, and TSLA.
  SEC EDGAR is the primary candidate and validation route because it provides
  official 8-K metadata, Item 2.02 tags, acceptance timestamps, and filing text.
  Massive 8-K text remains auxiliary fallback. Nasdaq calendar rows are not
  reliable enough as the primary historical timing source in the current probe
  because the matched SEC-confirmed sample had zero known Nasdaq timing flags.
- Earnings calendar builder: `build-earnings-calendar` now creates SEC-first
  candidate tables and validates accessions against SEC primary filing text by
  default, with Massive 8-K text available only as fallback. A live
  AAPL/MSFT/TSLA sample for 2026-01-01 through 2026-04-30 wrote
  `artifacts/earnings_calendar_sample/`: 8 SEC Item 2.02 candidates and 5
  main-sample candidates after timing and text validation.
- Active protocol file: `SPEC.md`.

Not yet paper evidence:

- A provisional top-50 pilot panel exists through `just data pilot-panel`; it is
  explicitly marked `provisional_no_nbbo` because it uses `options_day_aggs`
  close prices rather than pre-event NBBO quote pools.
- A V1.5 trade-price proxy route now exists through
  `just data trade-proxy-panel`. It uses Massive option second aggregates to
  take the latest pre-cutoff VWAP or close for candidate contracts, recompute
  local IV, and write a `no_nbbo_trade_proxy` event panel plus gross/haircut
  straddle diagnostics. Exit diagnostics use exit-date option day-aggregate
  closes by default and record `option_exit_price_status` plus
  `used_intrinsic_fallback` when intrinsic payoff is needed. This is useful for
  screening, not paper-grade execution claims.
- No paper-grade Massive `options_quotes_v1` ingestion has been implemented.
  The final top-50 proxy pipeline is implemented, but the full 2013-2025 data
  lake and paper-facing result tables/figures have not yet been produced.
- The active implementation is still deterministic/pilot plumbing: event
  alignment, variance extraction, leakage checks, data audit, contract
  discovery, local-IV diagnostics, and backtest smoke paths. It is not yet a
  train/test modeling pipeline.
- Integrity guards now fail closed on model-implementation claims, IVAR expiry
  coverage, timezone mismatches, missing/duplicate event-price rows, and
  vendor/local IV audit prerequisites.
- Contract discovery now records multiplier, contract size, deliverable status,
  corporate-action flags, and `contract_discovery_status`; non-standard OCC
  contracts are excluded before proxy-price pooling.
- Event-panel scaffolding now records `forward_source`, `forward_price`,
  `atm_selection_method`, `american_forward_caveat_flag`, and
  `possible_preannouncement_or_prior_guidance`.
- IVAR extraction failures now keep selected raw IVs, DTEs, expiries, spreads,
  and `expiry_gap_days` for diagnosis.
- No models or backtests have been run.

Latest local pilot-panel run:

- Command: `just data args="--force --max-events 50 --jobs 4"`.
- Gold panel: `data/gold/event_panel/pilot_event_panel.parquet`.
- Rows: 50 events in the current local calibration slice.
- RVAR/IVAR coverage: 50/50 events with RVAR and 44/50 with provisional IVAR.
- Contract candidates: 1146 selected near-ATM contracts; 1146 local IV estimates
  solved.
- Limitation: every row is `panel_grade = provisional_no_nbbo`; this is an
  engineering panel, not empirical evidence for the paper.

Latest local trade-proxy run:

- Command:
  `just data args="--force --max-events 50 --jobs 4"`.
- Gold panel: `data/gold/event_panel/trade_proxy_event_panel.parquet`.
- Rows: 50 events.
- Contract proxy prices: 1125/1146 contracts had pre-cutoff second-aggregate
  prices and local IV estimates; 21 contracts had
  `no_trade_in_cutoff_window`.
- Trade-proxy IVAR coverage: 44/50 events.
- Gross proxy straddle diagnostics: 44 rows. Mean gross proxy PnL is about
  `-117.93` USD and mean haircut PnL is about `-234.17` USD in this
  screening slice.
- Bronze second-agg cache: 1146 contract partitions written under
  `data/bronze/massive/options_second_aggs/`.
- Limitation: every row is `panel_grade = no_nbbo_trade_proxy`; this is a
  screening panel based on trade-price OHLCV bars, not a bid/ask executable
  strategy result.

## Legacy Material

Copied modules, scripts, data, generated site artifacts, and caches from earlier
projects have been removed from the active tree.

The current rule is:

- Active code must live under `src/earnings_event_vol`.
- Active tests must be migrated into the new test surface.
- Old copied code should be removed or explicitly migrated before being cited in
  docs or manuscript text.

## Next Gate

The next implementation gate is the uncapped 2013-2025 dynamic top-50 proxy
pass, not quote/NBBO ingestion. The recommended sequence is:

1. Run bare `just data` and inspect
   `artifacts/data_pipeline/options_day_aggs_bulk/options_day_aggs_bulk_manifest.json`,
   `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json`,
   and `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json`
   for coverage, repair status, fallback usage, stale contracts, and IVAR
   failures.
2. Use `just data args="--max-events 10"` only when a smoke run is desired.
3. Use `just data args="--dry-run"` before the full fetch, then use real run
   telemetry to decide whether the full 2013-2025 proxy lake stays on WSL ext4
   or moves `DATA_DIR` to a larger NVMe/external path.
4. Only after the proxy lake is stable, build the feature matrix and model
   baselines. Paper-grade quote/NBBO ingestion remains a later route, not a
   current dependency for the proxy data-engineering pass.

Promotion criterion:

- a small, reproducible sample with at least one BMO and one AMC event;
- no direct key leakage;
- explicit timestamp and quote-date audit output;
- docs updated with exact artifact paths and known limitations.
