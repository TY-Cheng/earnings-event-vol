---
hide:
  - navigation
---

# Future Work

Keep the first paper narrow. The current repo has a working no-NBBO proxy data
and modeling package, but the headline paper still needs paper-grade data,
robustness, and inference.

## Before Paper Claims

Paper-grade market data:

- Acquire or enable historical bid/ask or NBBO-equivalent option data.
- Rebuild IVAR and strategy legs from quote/NBBO inputs rather than
  second-aggregate trade bars.
- Report full bid-ask crossing as the main execution assumption.
- Keep mid and haircut cases as sensitivity tables only.

Full sample:

- Extend from the current 2022-onward entitlement-backed proxy sample to the
  target 2013-2025 sample.
- Preserve monthly top-50 liquid single-name universe construction.
- Keep ETF, index, volatility, commodity trust, and other non-single-name
  symbols out of the main universe.
- Re-run dynamic calendar, event panel, trade panel, feature matrix, models,
  figures, and reports after the full data route is stable.

Robustness and inference:

- Promote the current cost-stress and clustered forecast-loss diagnostics into
  final tables only after the paper-grade data route is rebuilt.
- Re-run main DTE `5-14` and robustness DTE `3-21` samples separately.
- Add block bootstrap confidence intervals.
- Add model-confidence-set or SPA-style checks if many thresholds or models are
  compared.
- Cross-tab results by year, ticker, sector, VIX regime, liquidity bucket, and
  BMO/AMC timing.

Sequence route:

- Reduce sequence-selection risk before using any sequence model as a paper
  claim.
- Report sequence coverage, drop rate, and missingness by year and ticker.
- Keep mask-only and deterministic time-shuffle controls.
- Keep official `mamba-ssm` framed as a diagnostic unless a future run beats
  mask-only/time-shuffle controls and tabular tuned rows on the common-row
  bootstrap gate.

## Near-Term Engineering

- Keep `just data` and `just research` as the public command surface.
- Add explicit stale-artifact checks so docs cannot cite obsolete 50-event or
  3-event calibration runs after a larger proxy run exists.
- Promote key proxy report numbers into `docs/results_snapshot.md` from
  machine-readable artifacts.
- Add a small command that prints the current sample window, event count, IVAR
  coverage, model rows, and paper-grade flag.
- Keep stale-result checks for `tuning_trials.csv`,
  `tuning_selected_params.json`, and FT finite-prediction diagnostics so the
  curated snapshot cannot silently reintroduce old original-model rows into the
  current canonical tuned story.
- Keep the same-code `fe_v1_legacy` versus `fe_v2_sec_xbrl` ablation table
  current. The 2026-05-12 run is negative for FE V2, so future FE V2 changes
  should be reported as diagnostics unless they improve locked-test ranking and
  economics without touching test-driven selection.
- Track SEC CompanyFacts coverage, CIK misses, acceptance-time mapping, and
  fallback-filed usage as first-class data-quality diagnostics.
- Keep generated data under the external `DATA_DIR`, and keep reports and
  figures under ignored `artifacts/`, `reports/`, or `site/`.

## Deferred Extensions

Calendar straddles:

- Add only after long straddle and short iron fly backtests are stable.
- Treat as a relative-value strategy, not a pure event-variance bet.
- Vega-normalize at entry and report residual Greeks.

Intraday execution:

- Use OPRA or another intraday quote source only after the EOD quote/NBBO design
  is audited.
- Study 15:45 or 15:59 entry, open-auction exit, and first 5/30/60 minute IV
  crush dynamics.

Richer event calendars:

- Add vendor calendars only when timestamp disagreements are auditable.
- Preserve BMO/AMC as the first paper sample.
- Keep DMH excluded unless the execution design becomes intraday.

Portfolio construction:

- Add volatility-budgeted allocation.
- Cap ticker, sector, and earnings-date concentration.
- Report capital-at-risk and premium-at-risk separately.

## Do Not Add Yet

- Naked short straddles.
- Unbounded short-gamma strategies.
- Hand-repaired earnings timestamps.
- Vendor proprietary alpha features that cannot be separated from model leakage.
- Claims based only on IV RMSE.
- Claims that no-NBBO proxy PnL is full-spread executable performance.
