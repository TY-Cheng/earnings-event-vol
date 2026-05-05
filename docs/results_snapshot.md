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
  57 tests passed with 93.55% total coverage.
- Data-audit gate: `just audit` for fixtures; `just audit date=YYYY-MM-DD` for
  a narrow Massive flat-file sample gate.
- Data-engineering gate: `just data` for resumable stages. The current stages
  are `fixture-audit`, `massive-probe`, `universe`, `calendar-pilot`,
  `contracts`, `panel`, `pilot-panel`, `trade-proxy-panel`, and `proxy-all`;
  outputs are skipped when present unless `--force` is passed through `args`.
- `just data` defaults to `proxy-all`, which runs
  `calendar-pilot -> pilot-panel -> trade-proxy-panel` over the Phase 1
  `2020-2025` range with a 10-event smoke cap, 4 workers, a 900-second pricing
  lookback, a 60-minute resolved-close pre-cutoff buffer, and DTE `3-21`
  contract discovery. Explicit stage names are still available for rebuilding
  the calendar, pilot panel, fixture audit, and legacy fixture panel smoke
  paths. It prints stage-level progress plus second-agg and exit day-agg
  count/status updates during long runs.
- Data lake layout: Massive flat-file downloads are temporary transfer files.
  They are converted immediately into compressed Parquet under `data/bronze/`;
  second-aggregate trade-proxy bars are cached under
  `data/bronze/massive/options_second_aggs/` for entry/pre-cutoff diagnostics
  only; exit option prices come from exit-date `options_day_aggs` closes.
  Cached Parquet partitions are reused if readable with the expected schema;
  corrupt second-agg or exit day-agg caches are repaired by deleting and
  re-fetching the affected partition.
  Cleaned intermediate tables go to `data/silver/`; analysis-ready panels go to
  `data/gold/`. `artifacts/` is reserved for manifests, readiness reports, and
  audit summaries.
- Dynamic universe scaffolding now exists as `just data universe
  args="--options-day-aggs PATH"`: it builds monthly ticker liquidity from
  normalized option day aggregates and top-50 trailing six-month option premium
  dollar-volume snapshots, with Phase 1 telemetry split into `covid_shock` and
  `steady_proxy`.
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
  compares Nasdaq calendar rows, SEC EDGAR company submissions, and Massive
  8-K text for AAPL, AMZN, MSFT, NVDA, and TSLA. SEC EDGAR submissions are the
  most reliable primary candidate generator tested so far because they provide
  official 8-K metadata, Item 2.02 tags, and SEC acceptance timestamps. Massive
  8-K text is used as the text-validation layer. Nasdaq calendar rows are not
  reliable enough as the primary historical timing source in the current probe
  because the matched SEC-confirmed sample had zero known Nasdaq timing flags.
- Earnings calendar builder: `build-earnings-calendar` now creates SEC-first
  candidate tables and can validate accessions against Massive 8-K text. A live
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
- No paper-grade Massive `options_quotes_v1` ingestion or final top-50 earnings
  panel has been implemented yet.
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

- Command: `just data pilot-panel args="--force --max-events 10"`.
- Gold panel: `data/gold/event_panel/pilot_event_panel.parquet`.
- Rows: 10 events, all AMC in the current local pilot slice.
- RVAR/IVAR coverage: 10/10 events with RVAR and 10/10 with provisional IVAR.
- Contract candidates: 180 selected near-ATM contracts; 180 local IV estimates
  solved.
- Limitation: every row is `panel_grade = provisional_no_nbbo`; this is an
  engineering panel, not empirical evidence for the paper.

Latest local trade-proxy run:

- Command:
  `just data args="--force --max-events 10 --jobs 4"`.
- Gold panel: `data/gold/event_panel/trade_proxy_event_panel.parquet`.
- Rows: 10 events.
- Contract proxy prices: 239/240 contracts had pre-cutoff second-aggregate
  prices and local IV estimates; 1 contract had
  `no_trade_in_cutoff_window`.
- Trade-proxy IVAR coverage: 10/10 events.
- Gross proxy straddle diagnostics: 10 rows. Mean gross proxy PnL is about
  `-268.61` USD and mean haircut PnL is about `-370.37` USD in this tiny
  screening slice.
- Bronze second-agg cache: 240 contract partitions written under
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

The next implementation gate is larger trade-proxy calibration, not quote/NBBO
ingestion. The recommended sequence is:

1. Run `just data args="--force --max-events 50 --jobs 4"` and inspect
   `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json` for
   coverage, repair status, fallback usage, stale contracts, and IVAR failures.
2. If coverage is healthy, run `just data args="--force --max-events 200 --jobs
   4"` to estimate Phase 1 storage/API/coverage telemetry.
3. Use that telemetry to decide whether the full 2020-2025 Phase 1 proxy lake
   stays on WSL ext4 or moves `DATA_DIR` to a larger NVMe/external path.
4. Only after the proxy lake is stable, build the feature matrix and model
   baselines. Paper-grade quote/NBBO ingestion remains a later route, not a
   current dependency for the proxy data-engineering pass.

Promotion criterion:

- a small, reproducible sample with at least one BMO and one AMC event;
- no direct key leakage;
- explicit timestamp and quote-date audit output;
- docs updated with exact artifact paths and known limitations.
