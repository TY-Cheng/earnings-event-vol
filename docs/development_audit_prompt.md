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
- `just status` is a lightweight environment diagnostic for resolved local
  paths and secret-file configuration; it is not a data/research rebuild.
- `just check` remains the full handoff gate.
- The test gate enforces at least 95% coverage.

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
- Market-data inputs are separated correctly: options day aggregates for
  universe/contract/IV proxy/exit/sequence construction, underlying day
  aggregates for vendor OHLC opens, C2O/C2C/O2C targets, and exit spot, and
  targeted option one-second aggregates for entry proxy pricing.
- Second-aggregate entry cache keeps only the pre-cutoff buffer, default 60
  minutes before event cutoff; entry price selection uses the true per-leg
  volume-weighted option VWAP over the final 900 seconds before cutoff.
- The option open anchor is unified as trade-aggregate 5-15 minute post-open
  VWAP. It is the primary C2O exit proxy and the O2C diagnostic entry proxy;
  0-5 minute VWAP remains only an opening microstructure stress test.
- O2C proxy PnL is diagnostic only unless a post-open residual-IV baseline is
  added.
- C2C exit diagnostics use same-contract option VWAP over the final 15 minutes
  before the exit-date close as the primary mark. Same-contract option
  day-aggregate close is not a strategy-exit fallback. Intrinsic fallback is
  flagged when the exit-preclose trade-aggregate mark is missing/unusable or
  when the option expires on the exit date.
- The current main no-NBBO target window is `2016-10-01` to `2026-06-05`.
  Broader Mac preflight artifacts from the previous `2016-01-01` window must
  be identified as preflight only. Current model/report metric artifacts may
  still be historical until rerun on the refreshed main-window feature matrix.
- Universe construction filters ETF/index/non-single-name symbols before
  selecting the monthly top 50.
- SEC EDGAR submissions plus SEC primary filing documents are the primary
  earnings candidate and text-validation route.
- Massive 8-K text is auxiliary fallback only.

Event variance construction:

- AMC and BMO event windows use the documented pre-announcement close.
- DMH and unknown timing are excluded from the first main sample.
- Target construction writes C2O, C2C, and O2C realized-variance columns.
- `rvar_event` remains the documented close-to-close alias for current proxy PnL.
- C2O is not described as executable option PnL.
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
  LightGBM/XGBoost, FT-Transformer, and sequence diagnostics are compared only
  when their callable implementations and diagnostics exist.
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
