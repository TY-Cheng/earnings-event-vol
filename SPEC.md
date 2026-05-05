# Earnings Event Vol V1 Implementation Spec

This repository implements a reproducible research skeleton for earnings event
variance forecasting and risk-defined options backtests. Models forecast
`RVAR_event`; ex post mispricing evaluates correctness; trading decisions use
premium-space expected edge, not raw variance edge.

## Protocol Defaults

- Main event move is close-to-close:
  - AMC: `S_before = close_d`, `S_after = close_{d+1}`.
  - BMO: `S_before = close_{d-1}`, `S_after = close_d`.
- `IVAR_event` uses two adjacent expiries covering the event:
  - `T_j` follows the IV source year-fraction convention.
  - V1 default is ACT/365 unless vendor documentation says otherwise.
  - `w_j = IV_j^2 * T_j`.
  - `IVAR_event = (T2*w1 - T1*w2) / (T2 - T1)`.
  - Failure reports keep selected raw IVs, DTEs, expiries, spreads, and
    `expiry_gap_days`.
- Contract discovery must run before quote pooling. Primary-sample contracts
  require `option_multiplier == 100`, `contract_size == 100`, and
  `deliverable_status == standard`; non-standard OCC deliverables are reported
  as `non_standard_excluded`.
- V1.5 may build a trade-price proxy panel from Massive option second
  aggregates:
  - For each candidate contract, use the latest pre-cutoff VWAP or close inside
    the configured lookback window.
  - Require paired call/put proxy prices before constructing an expiry-level ATM
    IV input.
  - Mark all outputs `no_nbbo_trade_proxy`.
  - Report gross proxy PnL and haircut PnL only as screening diagnostics.
  - Do not use this route for full-spread crossing or paper-grade execution
    claims.
- ATMF forward selection can use put-call parity only as a short-DTE,
  no-dividend, near-ATM approximation for American single-name options. Weak or
  dividend-contaminated pairs fall back to nearest-spot ATM and record
  `forward_source = spot_fallback`.
- Raw research edge is `edge_var = forecast_RVAR_event - IVAR_event`.
- Trading threshold is premium-space:
  - Convert forecast variance into expected strategy value using deterministic
    quadrature under a zero-mean Gaussian event-return distribution.
  - The current v1 smoke valuation marks each leg to intrinsic payoff after the
    event jump and ignores residual post-event time value, discounting, and
    post-event implied-volatility repricing.
  - `expected_strategy_edge_usd = expected_strategy_value_usd - market_entry_cost_usd`.
  - Primary rule: `expected_strategy_edge_usd > 1.5 * estimated_transaction_cost_usd`.
- Every feature row must satisfy `feature_asof_timestamp <= event_entry_timestamp`.
- Main sizing uses fractional theoretical contracts; integer-contract PnL is
  reported as tradability robustness.

## V1 Boundaries

- Massive is the v1 data route, subject to field audit.
- Earnings calendar input must provide explicit BMO/AMC/DMH/UNKNOWN timing.
  The active candidate route is SEC EDGAR 8-K Item 2.02 metadata plus Massive
  8-K text validation; unresolved or ambiguous timing stays outside the main
  sample.
- No inferred timestamp fallback.
- No calendar spread, intraday simulator, surface projection, GNN/GNO, or full
  deep-model training in this pass.
