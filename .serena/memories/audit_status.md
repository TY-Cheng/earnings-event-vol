# Audit Status

Last synced: 2026-06-13 after switching the main target window to
2016-10-01 through 2026-06-05.

## Current Verification

- Current Mac checkout root is
  `/Users/tycheng/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/earnings-event-vol/earnings-event-vol`;
  other machines should use their active checkout path and machine-local `.env`.
- Full `just check` passed on 2026-06-13 after the latest
  contract-reference, manifest-canonicalization, rate-limiter, feature, docs,
  and research-helper fixes: 175 tests passed, total coverage 95.13%, ruff
  format/check passed, mypy passed, MkDocs strict build passed, CLI `status`
  passed, and `source-probe all` passed.
- Remaining warnings are upstream/dependency warnings; they are non-blocking.

## Current Artifact Audit

- The Mac checkout has a broader pre-window-change preflight rebuild for
  2016-01-01 through 2026-06-05: options day aggregates, universe, dynamic SEC
  calendar, SEC CompanyFacts, event-window/contract-candidate setup,
  contract-reference validation, trade-proxy panel, and the gold feature matrix
  were refreshed. Do not cite those counts as current main-window evidence until
  the 2016-10-01 through 2026-06-05 pipeline is rerun.
- Preflight event-window report has 3,072 events, 3,001 events with realized
  variance, 3,071 events with entry-window support, 80,275 contract candidates,
  40,709 quote-pool contracts, 17,595 main DTE 5-14 contracts, and 39,566
  IVAR-support-only contracts.
- Preflight SEC CompanyFacts manifest has 228,205 standardized fact rows for
  201 tickers.
- Preflight contract-reference validation covers 79,903 unique tickers: 79,634
  validated and 269 `missing_reference`; unknown deliverables are excluded from
  proxy usability.
- Preflight trade-proxy panel has 3,072 events, 3,001 events with RVAR, 2,538
  events with trade-proxy IVAR, 80,006 proxy-usable contract rows, 55,580
  contracts with usable pre-entry trade proxy marks, and 24,426 with no trade
  in the cutoff window.
- Preflight feature matrix has 3,071 rows, 559 columns, and 415 model features
  under `fe_v2_sec_xbrl`; low-dimensional additions include sequence call/put
  volume imbalance aggregates, own-underlying pre-event return/RV run-up, and
  SEC SIC coarse controls.
- Preflight lake-quality audit is `ok=false`: options day aggregates are
  covered, underlying day aggregates start on 2016-06-13 under the current
  Massive entitlement, and full bid/ask/NBBO-equivalent quote coverage is still
  missing. The main 2016-10-01 window starts after the known 2016-H1 underlying
  entitlement gap.
- Current model/report outputs have not yet been rerun against a refreshed
  2016-10-01 through 2026-06-05 feature matrix.

## Current Docs Audit

- `docs/paper_plan.md` is the paper-style manuscript plan.
- `docs/results_snapshot.md` is the paper-style Results and Discussion ledger.
- README, paper plan, results snapshot, and current Serena memories distinguish
  the broader 2016-01-01 preflight data layers from the active 2016-10-01 main
  window and from the old 816-row historical modeling snapshot.
- The PR #1 Chinese contributor docs are no longer present in `docs/` or
  MkDocs nav; their useful ideas were absorbed into paper-facing docs and code.

## Current Legacy Audit

- The retired legacy feature schema has been removed from the command/config/code
  surface. The only accepted feature schema is `fe_v2_sec_xbrl`.
- Retired legacy sequence ids remain only in retirement manifests/tests and are
  not active model ids or runtime dependencies.
- Historical ablation artifacts may remain under ignored
  `artifacts/modeling_ablations/`; treat them as stale local artifacts only.
- The current result set is not final paper-grade evidence because full
  bid/ask/NBBO-equivalent execution remains pending and model/report outputs
  need a rerun on the refreshed 2016-10-01 through 2026-06-05 feature matrix.
