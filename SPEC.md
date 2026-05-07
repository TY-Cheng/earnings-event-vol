# Earnings Event Vol V1 Implementation Spec

This repository implements a reproducible research skeleton for earnings event
variance forecasting and risk-defined options backtests. Models forecast
realized earnings-event variance targets; ex post mispricing evaluates
correctness; trading decisions use premium-space expected edge, not raw
variance edge.

## Protocol Defaults

- Final proxy data-engineering entrypoint:
  - The public command remains `just data` / `proxy-all`.
  - `proxy-all` runs `options-day-aggs-bulk -> universe ->
    dynamic-calendar -> pilot-panel -> contract-reference-validation ->
    trade-proxy-panel`.
  - The runnable Massive-entitlement default is 2022-12-01 through
    2025-12-31. Universe construction downloads the trailing six-month lookback
    beginning 2022-06-01 for the December 2022 universe.
  - The paper target remains 2013-2025, but the current Massive flat-file
    entitlement observed in this workspace exposes option day aggregates only
    from 2022-05-04. Earlier dynamic-universe runs require upgraded historical
    options day-agg entitlement or another licensed options data route.
  - Universe construction must filter to eligible SEC company/common-equity
    tickers before the top-50 ranking. ETF, index, volatility, commodity trust,
    and other non-single-name symbols such as SPX, SPXW, SPY, QQQ, IWM, VIX,
    and GLD must not consume top-50 slots. The eligibility cache records source,
    snapshot date, rule version, exchange/name filter reason, and cache
    invalidation diagnostics.
  - `calendar-pilot` is a static ticker smoke/debug stage and is not part of the
    default final proxy DAG.
- Event targets are decomposed:
  - Primary scientific target: `RVAR_event_jump_c2o`.
    - AMC: `close_d -> open_{d+1}`.
    - BMO: `close_{d-1} -> open_d`.
  - Literature-compatible and V1 proxy-PnL target: `RVAR_event_day_c2c`.
    - AMC: `close_d -> close_{d+1}`.
    - BMO: `close_{d-1} -> close_d`.
  - Diagnostic post-open digestion target: `RVAR_event_reaction_o2c`.
    - AMC: `open_{d+1} -> close_{d+1}`.
    - BMO: `open_d -> close_d`.
  - `rvar_event` remains a backward-compatible alias for
    `RVAR_event_day_c2c`.
  - The return identity is exact:
    `r_event_day_c2c = r_event_jump_c2o + r_event_reaction_o2c`.
    Variance reconstruction must include
    `RVAR_cross_term = 2 * r_event_jump_c2o * r_event_reaction_o2c`.
  - Open prices from Massive daily OHLC may be used as
    `vendor_regular_ohlc_assumed`; they are not verified auction-open prints.
- `IVAR_event` uses two adjacent expiries covering the event:
  - `T_j` follows the IV source year-fraction convention.
  - V1 default is ACT/365 unless vendor documentation says otherwise.
  - `w_j = IV_j^2 * T_j`.
  - `IVAR_event = (T2*w1 - T1*w2) / (T2 - T1)`.
  - Failure reports keep selected raw IVs, DTEs, expiries, spreads, and
    `expiry_gap_days`.
- Contract discovery must run before proxy-price or paper-grade quote pooling.
  Primary-sample contracts require `option_multiplier == 100`,
  `contract_size == 100`, and `deliverable_status == standard`; non-standard
  OCC deliverables are reported as `non_standard_excluded`.
- `contract-reference-validation` validates selected candidate option contracts
  against Massive `/v3/reference/options/contracts` metadata before
  second-aggregate entry fetching:
  - Read `shares_per_contract`, `additional_underlyings`, `exercise_style`, and
    `correction`.
  - Override `option_multiplier`, `contract_size`, and `deliverable_status`
    when reference metadata is available.
  - Exclude contracts with non-100 `shares_per_contract` or adjusted
    deliverables as `non_standard_excluded`.
  - Reference-fetch failures are manifest diagnostics only. They do not make a
    contract paper-grade and do not create bid/ask or NBBO claims.
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
- SPY/QQQ market-state controls are required when available in the existing
  data lake; when unavailable, coverage must be reported and the no-market-control
  specification remains valid. They use the same proxy discipline:
  - `market-second-covariates` fetches SPY and QQQ option one-second aggregates
    plus SPY/QQQ underlying one-second aggregates at the event entry cutoff.
  - It produces event-level ATM IV, term slope, skew, butterfly,
    straddle-premium-to-spot, option activity, and underlying pre-cutoff return
    proxies.
  - These features are trade OHLCV aggregates, not bid/ask, quote, or NBBO
    surfaces.
- The daily Mamba path is daily-frequency. Each timestep is one allowed
  pre-entry trading day with close-trade-implied single-name option-surface
  summaries, SPY/QQQ daily surface summaries when available, market ETF returns,
  and daily VIX state.
- The hybrid proxy sequence path uses 31 timesteps:
  - steps 00-18 are the prior 19 trading days before the entry date;
  - steps 19-30 are twelve entry-day five-minute bins from cutoff minus
    60 minutes through cutoff;
  - entry-date daily close is excluded from the daily segment;
  - `step_type`, `is_intraday_bin`, `log_delta_minutes_from_prev_step`,
    `normalized_time_to_entry`, `hours_until_announcement_proxy`, and
    `iv_extraction_source` distinguish mixed-frequency observations;
  - SEC acceptance time is an announcement proxy, not verified first-release
    time;
  - the intraday sequence is a trade-aggregate proxy surface from second
    aggregate OHLCV bars, not a quote/NBBO surface;
  - if fewer than 70% of events have at least eight valid intraday bins, or
    median hybrid mask density is below 0.50, hybrid Mamba results are labeled
    `high_missingness_diagnostic` and cannot be headline evidence.
- ATMF forward selection can use put-call parity only as a short-DTE,
  no-dividend, near-ATM approximation for American single-name options. Weak or
  dividend-contaminated pairs fall back to nearest-spot ATM and record
  `forward_source = spot_fallback`.
- Raw V1 strategy edge is C2C-only:
  `edge_var_day_c2c = forecast_RVAR_event_day_c2c - IVAR_event`. C2O is
  reported as forecast/ranking of realized jump variance, not as a V1 tradable
  mispricing or option-PnL headline.
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

## Market Covariate Alignment

VIX is a daily market-state covariate and regime control, not execution data.
The default source route is the public FRED graph CSV:
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS`. The raw copy is
cached as `data/bronze/market_covariates/fred_vixcls.csv`; normalized fields are
written to `data/silver/market_covariates/daily_market_covariates.parquet` with
`source_dataset`, `source_url`, `source_snapshot_date`, and `schema_version`.

The headline alignment is `vix_alignment = prior_close_default`.
`feature_asof_date` is the trading date whose close marks the event entry
timestamp: BMO maps to the trading day immediately before the announcement,
while AMC maps to the announcement trading date. Headline VIX lookup requires
`vix_date < feature_asof_date` for both BMO and AMC, with
`max_vix_lag_days = 5`. Same-day VIX may be allowed only in the
`same_day_close_for_amc` robustness specification, where AMC permits
`vix_date <= feature_asof_date` and BMO remains strict prior-close.

VIX feature construction resolves `resolved_vix_date` first. Changes use valid
VIX observations rather than calendar-day offsets:

```text
vix_change_1d = VIX(resolved_vix_date) - VIX(previous valid VIX date)
vix_change_5d = VIX(resolved_vix_date) - VIX(5th previous valid VIX observation)
```

`vix_percentile_252d` and `vix_regime_tercile` use only valid observations with
`vix_date < resolved_vix_date`; the resolved date itself is excluded from the
rolling cutpoint history. Regime construction requires at least 40 prior valid
observations. `vix_above_30` may use the resolved VIX value directly because it
does not estimate a rolling cutpoint.

## V1 Boundaries

- Massive is the v1 data route, subject to field audit.
- Earnings calendar input must provide explicit BMO/AMC/DMH/UNKNOWN timing.
  The active candidate route is SEC EDGAR 8-K Item 2.02 metadata plus SEC
  primary filing document text validation. Massive 8-K text may be used only as
  auxiliary fallback when official SEC document text is unavailable or
  inconclusive; it is not a required calendar dependency. Unresolved or
  ambiguous timing stays outside the main sample.
- No inferred timestamp fallback.
- No calendar spread, intraday simulator, surface projection, GNN/GNO, or
  paper-grade deep-model tuning claims in this pass. The proxy research layer
  may train the registered benchmark/model suite for diagnostics, but outputs
  remain `no_nbbo_trade_proxy` until quote/NBBO ingestion exists.
