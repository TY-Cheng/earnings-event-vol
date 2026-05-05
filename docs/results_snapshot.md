# Results Snapshot

## Current State

This repository is an early-stage research pipeline for the earnings event
volatility project. The current implementation can build audited candidate
events, provisional event panels, and a second-aggregate trade-price proxy
panel. It does not yet produce paper-grade quote/NBBO backtest results.

Active and current:

- Project name: `earnings-event-vol`.
- Python package: `src/earnings_event_vol`.
- Local environment: `UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/earnings-event-vol"`.
- Credential policy: Massive keys are file-only through `.env`.
- Verification front door: `just check`, including `ruff format`, `ruff check
  --fix`, strict mypy, pytest, MkDocs strict, status, and Massive credential
  probes.
- Test coverage floor: 93% on the active package. Current local gate:
  51 tests passed with 94.23% total coverage.
- Data-audit gate: `just audit` for fixtures; `just audit date=YYYY-MM-DD` for
  a narrow Massive flat-file sample gate.
- Data-engineering gate: `just data` for resumable stages. The current stages
  are `fixture-audit`, `massive-probe`, `calendar-pilot`, `contracts`,
  `panel`, `pilot-panel`, and `trade-proxy-panel`; outputs are skipped when
  present unless `--force` is passed through `args`.
- `just data` defaults to the `trade-proxy-panel` stage with a 10-event,
  4-worker, 900-second pre-cutoff VWAP window. Explicit stage names are still
  available for rebuilding the calendar, pilot panel, fixture audit, and quote
  panel smoke paths.
- Data lake layout: Massive downloads are temporary transfer files. They are
  converted immediately into compressed Parquet under `data/bronze/`; cleaned
  intermediate tables go to `data/silver/`; analysis-ready panels go to
  `data/gold/`. `artifacts/` is reserved for manifests, readiness reports, and
  audit summaries.
- First real Massive flat-file probe: `just audit date=2025-02-05` succeeded
  for option day aggregates, option quotes metadata, and underlying day
  aggregates. Outputs are in `artifacts/massive_flat_file_probe/`.
- Current data-readiness finding: `day_aggs_v1` supports contract parsing,
  underlying close sampling, and a provisional local-IV route from option close
  prices. It lacks bid/ask, open interest, quote condition, IV, and Greeks, so
  it cannot support paper-grade IVAR extraction or transaction-cost backtests by
  itself. The next data route must use `options_quotes_v1` plus an IV/Greeks/OI
  source or a local IV solver from NBBO mids.
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
  straddle diagnostics. This is useful for screening, not paper-grade execution
  claims.
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
  contracts are excluded before quote pooling.
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
  `just data trade-proxy-panel args="--force --max-events 1 --max-contracts 20 --jobs 4 --lookback-seconds 900"`.
- Gold panel: `data/gold/event_panel/trade_proxy_event_panel.parquet`.
- Rows: 1 AMC event, with `trade_proxy_ivar_event` successfully extracted.
- Contract proxy prices: 18/18 contracts had pre-cutoff second-aggregate prices.
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

The next implementation gate is trade-proxy screening, then paper-grade
quote/IV ingestion:

1. Run `just data trade-proxy-panel` on a small recent event sample and inspect
   `trade_proxy_panel_report.json` for coverage, stale contracts, and IVAR
   failures.
2. Decide how to extract end-of-day option quotes from `options_quotes_v1`
   without downloading a full 95GB day file into memory.
3. Add an IV/OI path: either Massive historical snapshot/contract endpoint if
   available for the date, or local IV calculation from quote mid plus contract
   metadata and underlying close.
4. Replace the provisional `options_day_aggs` close proxy in
   `data/gold/event_panel/pilot_event_panel.parquet` with pre-event NBBO quote
   pools and quote-condition/liquidity diagnostics.
5. Produce an event-alignment audit showing no post-announcement quotes enter
   predictors.
6. Promote a real data audit only after required Massive fields, quote-source
   semantics, vendor/local IV differences, and earnings timestamp source are
   documented.

Promotion criterion:

- a small, reproducible sample with at least one BMO and one AMC event;
- no direct key leakage;
- explicit timestamp and quote-date audit output;
- docs updated with exact artifact paths and known limitations.
