---
hide:
  - navigation
---

# Future Work

Keep the first paper narrow. The current repo has a completed no-NBBO proxy
cold run for 2016-10-01 to 2026-06-05, including data, features, models,
reports, quote-aware diagnostics, IVAR defeat analysis, and casebook outputs.
The headline paper still needs paper-grade execution data and final robustness
discipline before making executable trading claims.

## Before Paper Claims

Paper-grade market data:

- Acquire or enable historical bid/ask or NBBO-equivalent option data.
- Rebuild IVAR and strategy legs from quote/NBBO inputs rather than
  second-aggregate trade bars.
- Report full bid-ask crossing as the main execution assumption.
- Keep mid and haircut cases as sensitivity tables only.

Full sample:

- Treat the 2016-10-01 to 2026-06-05 no-NBBO cold run as the current proxy
  baseline.
- Do not cite older repo-local artifacts, older bounded 502-event quote slices,
  or broader preflight materializations as current evidence.
- Preserve monthly top-50 liquid single-name universe construction.
- Keep ETF, index, volatility, commodity trust, and other non-single-name
  symbols out of the main universe.
- Re-run models, figures, reports, and robustness tables only after a new data
  or paper-grade quote/NBBO route changes the evidence base.

Robustness and inference:

- Promote the current cost-stress and clustered forecast-loss diagnostics into
  final tables only after the paper-grade data route is complete.
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
- Keep sequence models framed as diagnostics unless a future run beats
  mask-only/time-shuffle controls and tabular tuned rows on the common-row
  bootstrap gate.

## Near-Term Engineering

- Keep `just data` and `just research` as the public command surface.
- Add explicit stale-artifact checks so docs cannot cite obsolete 50-event or
  3-event calibration runs after a larger proxy run exists.
- Promote key proxy report numbers into `docs/results_snapshot.md` from
  machine-readable artifacts whenever the external cold-run root changes.
- Add a small command that prints the current sample window, event count, IVAR
  coverage, model rows, and paper-grade flag.
- Keep stale-result checks for `tuning_trials.csv`,
  `tuning_selected_params.json`, and FT finite-prediction diagnostics so the
  curated snapshot cannot silently reintroduce old original-model rows into the
  current canonical tuned story.
- Keep `feature_schema_report.csv` and the leakage audit synchronized with any
  future feature changes; judge changes by validation/test ranking and
  premium-space economics without touching test-driven selection.
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
