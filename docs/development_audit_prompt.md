# Development Audit Brief

Audit `earnings-event-vol` as an earnings option event-variance research
codebase.

## Binding Scope

The active project tests whether models improve trading decisions around
option-implied earnings event variance mispricing.

Do not audit it as:

- a generic implied-volatility forecasting project;
- a generic options-pricing surface reconstruction project;
- a paper-grade NBBO backtest already.

The current implemented route is a no-NBBO trade-price proxy. It is useful for
pipeline validation and signal screening, not final execution claims.

## Required Checks

Project identity and workflow:

- `pyproject.toml` uses `earnings-event-vol`.
- Active package is `src/earnings_event_vol`.
- `justfile` calls `python -m earnings_event_vol.cli`.
- Public command surface stays small: `status`, `audit`, `data`, `research`,
  `docs`, and `check`.
- `just check` remains the handoff gate.
- The test gate enforces at least 93% coverage.

Credential safety:

- Massive secrets are file-only.
- `MASSIVE_API_KEY_FILE` and `MASSIVE_FLAT_FILE_KEY_FILE` point to local secret
  files.
- Source probes never print secret values.
- Direct API-key values are not accepted as the primary path in source, docs, or
  tests.

Data route and labeling:

- Current data outputs are labeled `no_nbbo_trade_proxy` and `paper_grade=false`.
- Second aggregates are described as trade-price OHLCV bars, not bid/ask or
  NBBO quotes.
- Exit diagnostics use same-contract option day-aggregate closes when available
  and flag intrinsic fallback use.
- The current default proxy range is the observed entitlement range
  `2022-12-01` to `2025-12-31`; 2013-2025 remains the target paper range, not
  the current completed result.
- Universe construction filters ETF/index/non-single-name symbols before
  selecting the monthly top 50.
- SEC EDGAR submissions plus SEC primary filing documents are the primary
  earnings candidate and text-validation route.
- Massive 8-K text is auxiliary fallback only.

Event variance construction:

- AMC and BMO event windows use the documented pre-announcement close.
- DMH and unknown timing are excluded from the first main sample.
- `RVAR_event` uses the documented close-to-close event move.
- `IVAR_event` is extracted from total ATM implied variance across two expiries
  that cover the realized event window.
- Negative and nonmonotone IVAR extractions are flagged and reported.
- DTE losses and IVAR failures are surfaced in manifests and reports.

Leakage and timestamp gates:

- Every feature row has an as-of timestamp at or before event entry.
- Timezones are explicit and consistent; naive/aware mixtures fail closed.
- Same-event realized fields, post-event fields, and vendor forecast leakage are
  excluded unless explicitly whitelisted for diagnostics.
- Temporal splits are chronological or walk-forward, not random.

Model and research package:

- Market-implied IVAR is always the primary benchmark.
- Last-four RVAR, last-four IVAR, Goyal-Saretto-style RV-IV spread, Elastic Net,
  LightGBM/XGBoost, FT-Transformer, and Mamba are compared only when their
  callable implementations and diagnostics exist.
- Model registry `implemented` flags must match callable behavior.
- Sequence-model results report coverage, drop rate, and mask-only ablation.
- High sequence-selection risk is surfaced and not hidden in headline claims.
- Forecast metrics, ranking metrics, strategy metrics, cost sensitivity,
  inference diagnostics, and model-fit diagnostics are written under
  `artifacts/modeling/`.

Backtest and execution claims:

- Proxy strategy summaries are cost-aware screening diagnostics.
- Full bid-ask crossing or NBBO execution is a future paper-grade requirement,
  not a claim supported by the current proxy route.
- Long ATM straddle and short iron fly remain the v1 headline strategy designs.
- Calendar spread is labeled as second-stage relative value.
- Multi-leg fills document simultaneous-fill assumptions and legging-risk
  limitations.

## Output

Report findings in this order:

1. Blockers.
2. Leakage or timestamp risks.
3. Data-source, entitlement, or credential risks.
4. Model and research-package completeness gaps.
5. Backtest and execution-claim gaps.
6. Documentation drift.
