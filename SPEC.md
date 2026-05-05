# Earnings Event Vol V1 Implementation Spec

This repository implements a reproducible research skeleton for earnings event
variance forecasting and risk-defined options backtests. Models forecast
`RVAR_event`; ex post mispricing evaluates correctness; trading decisions use
premium-space expected edge, not raw variance edge.

## Protocol Defaults

- Final proxy data-engineering entrypoint:
  - The public command remains `just data` / `proxy-all`.
  - `proxy-all` runs `options-day-aggs-bulk -> universe ->
    dynamic-calendar -> pilot-panel -> trade-proxy-panel`.
  - Study window defaults to 2013-01-01 through 2025-12-31.
  - Universe construction downloads the trailing six-month lookback beginning
    2012-07-01 for the January 2013 universe.
  - `calendar-pilot` is a static ticker smoke/debug stage and is not part of the
    default final proxy DAG.
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
- Contract discovery must run before proxy-price or paper-grade quote pooling.
  Primary-sample contracts
  require `option_multiplier == 100`, `contract_size == 100`, and
  `deliverable_status == standard`; non-standard OCC deliverables are reported
  as `non_standard_excluded`.
- V1.5 may build a trade-price proxy panel from Massive option second
  aggregates:
  - For each candidate contract, use the latest pre-cutoff VWAP or close inside
    the configured lookback window.
  - The second-aggregate bronze cache serves entry pricing and pre-cutoff
    feature/liquidity diagnostics only. Do not retain post-cutoff bars for this
    route.
  - The pre-cutoff window is anchored to a resolved market-close timestamp. The
    resolver must return timezone-aware America/New_York and UTC timestamps and
    must use early-close times, such as 13:00 ET, when supplied by the trading
    calendar.
  - Require paired call/put proxy prices before constructing an expiry-level ATM
    IV input.
  - Exit diagnostics use `underlying_exit_price_source = day_aggs_close`,
    `option_exit_price_source = options_day_aggs_close`, and
    `option_exit_payoff_fallback = intrinsic_value_at_underlying_exit`.
    Same-contract option day-aggregate close is the default exit mark; intrinsic
    payoff is used only when the exit option close is missing/unusable or when
    the option expires on the exit date.
  - Proxy PnL outputs must record `option_exit_price_source`,
    `option_exit_price_status`, and `used_intrinsic_fallback`.
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
  - The deterministic v1 backtest smoke remains a simplified payoff model. The
    V1.5 trade-proxy diagnostics preserve residual option extrinsic value when
    exit-date option day-aggregate closes are available.
  - `expected_strategy_edge_usd = expected_strategy_value_usd - market_entry_cost_usd`.
  - Primary rule: `expected_strategy_edge_usd > 1.5 * estimated_transaction_cost_usd`.
- Every feature row must satisfy `feature_asof_timestamp <= event_entry_timestamp`.
- Main sizing uses fractional theoretical contracts; integer-contract PnL is
  reported as tradability robustness.
- Ticker eligibility and mapping should prefer point-in-time vendor mappings
  when available. If a reliable PIT source is unavailable, use SEC current
  company metadata plus an options-chain reality check; missing or ambiguous
  mappings must be surfaced as manifest exclusions such as `ticker_not_found`,
  `ticker_mapping_ambiguous`, or `no_active_option_chain`, not guessed.
- External academic crosswalks such as GVKEY-CIK link tables may be used only as
  optional curated mapping inputs. They can connect SEC CIKs to Compustat-style
  firm identifiers, but they do not replace point-in-time ticker or options
  chain validation for the market-data route.
- SEC EDGAR event discovery must normalize both `filings.recent` and archived
  `filings.files` submission JSON. Archive fetch failures are diagnostics, not
  hard failures for the whole universe ticker set, and accessions are deduped
  across recent and archived payloads.

## V1 Boundaries

- Massive is the v1 data route, subject to field audit.
- Earnings calendar input must provide explicit BMO/AMC/DMH/UNKNOWN timing.
  The active candidate route is SEC EDGAR 8-K Item 2.02 metadata plus SEC
  primary filing document text validation. Massive 8-K text may be used only as
  auxiliary fallback when official SEC document text is unavailable or
  inconclusive; it is not a required calendar dependency. Unresolved or
  ambiguous timing stays outside the main sample.
- No inferred timestamp fallback.
- No calendar spread, intraday simulator, surface projection, GNN/GNO, or full
  deep-model training in this pass.
