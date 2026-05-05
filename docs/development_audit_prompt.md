# Development Audit Brief

Audit the `earnings-event-vol` repository as an earnings option event-variance
research codebase.

## Binding Scope

The active project is:

> A reproducible pipeline for testing whether ML/DL forecasts improve
> option-implied earnings event variance trading decisions.

Do not audit it as a generic IV-surface reconstruction project or as a copied
corporate-reporting pipeline.

## Required Checks

Project identity:

- `pyproject.toml` uses `earnings-event-vol`.
- Active package is `src/earnings_event_vol`.
- `justfile` calls `python -m earnings_event_vol.cli`.
- `.env.example` uses `${HOME}/.venvs/earnings-event-vol`.
- Docs and CI do not refer to old project names as active work.
- The active test gate enforces at least 93% coverage and is run through
  `just check`.

Credential safety:

- Massive secrets are file-only.
- Source probes never print secret values.
- Direct API-key environment variables are not accepted as the primary path.

Data readiness:

- Option and underlying samples are aligned by quote date and ticker.
- Earnings events distinguish BMO, AMC, DMH, and unknown.
- DMH and unknown are excluded in v1.
- AMC and BMO event windows use the correct pre-announcement close.

Event variance construction:

- `IVAR_event` is extracted from total ATM implied variance across two expiries.
- IVAR extraction uses expiries that cover the realized event window, not only
  the announcement date.
- Negative extractions are flagged and reported.
- `RVAR_event` uses the documented EOD event move definition.
- The code reports extraction failures and DTE filter losses.
- Trading entry thresholds compare USD expected strategy edge to USD transaction
  costs; raw variance edge is not compared directly to option spreads.
- Data audit outputs required field coverage, quote-source flags, and
  vendor-vs-local IV diagnostics before any backtest is promoted.
- Second-aggregate trade-proxy panels are labeled `no_nbbo_trade_proxy` and are
  not described as full-spread executable backtests.
- Leakage audit enforces feature as-of timestamps and blocks vendor forecast or
  same-event realized fields unless explicitly whitelisted.
- Event-entry and feature as-of timestamps are timezone-consistent; naive/aware
  mixtures fail closed.

Model and backtest gates:

- Market-implied event variance is always a baseline.
- Last-four earnings moves and LightGBM are included before deep models.
- Model registry implementation flags match callable implementations; planned
  baselines are not marked implemented before they exist.
- Backtests include full bid-ask crossing.
- Long straddle and short iron fly are the v1 headline strategies.
- Calendar spread is labeled as second-stage relative value.
- Multi-leg fills document the simultaneous-fill assumption and legging-risk
  limitation.

## Output

Report findings as:

1. Blockers.
2. Leakage or timestamp risks.
3. Data-source or credential risks.
4. Model/backtest completeness gaps.
5. Documentation drift.
