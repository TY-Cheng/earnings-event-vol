# Research Overview

This project studies a narrow question:

> Around U.S. earnings announcements, can models improve the selection of
> option trades after the market-implied event variance and bid-ask costs are
> already taken seriously?

The target is not a generic implied-volatility forecasting task. The trading
problem is event-specific. Before an earnings announcement, short-dated option
prices embed a view of the jump variance. After the announcement, the stock
realizes one event move. The study asks whether pre-event information can rank
which events are cheap or rich enough to trade.

## Core Objects

For each earnings event:

```text
RVAR_event = log(S_after / S_before)^2
```

`S_before` and `S_after` depend on announcement timing:

- AMC: enter before the close on the announcement date; measure close-to-close
  from `d` to `d+1`.
- BMO: enter before the close on the previous trading day; measure close-to-close
  from `d-1` to `d`.
- DMH and unknown timing are excluded from the first main sample.

The option-market benchmark is `IVAR_event`, extracted from two short-dated
expiries that both cover the realized event window. The raw research edge is:

```text
forecast_RVAR_event - IVAR_event
```

The trade decision is not made directly in variance units. The forecast is
translated into expected option-strategy value in dollars, then compared with
estimated transaction costs:

```text
expected_strategy_edge_usd > 1.5 * estimated_transaction_cost_usd
```

That distinction matters. A model can look better on RMSE and still be useless
if the signal does not survive bid-ask spreads.

ATM and forward selection are audited fields, not hidden implementation details.
The preferred route uses a liquid near-ATM call/put pair to estimate an ATMF
forward by put-call parity. Because these are American single-name options, that
forward is treated as a short-DTE, no-dividend approximation. If the pair is
weak or the event-to-second-expiry window crosses an ex-dividend date, the panel
falls back to nearest-spot ATM and records `forward_source = spot_fallback`.

## Data Route

Main study target:

- Top 50 liquid U.S. single-name option stocks.
- 2013-2025.
- BMO and AMC earnings events only.
- Main event-expiry DTE range: 5-14.
- Robustness DTE range: 3-21.

Market data:

- Massive option and stock flat files are the primary route for options and
  underlying prices.
- `day_aggs_v1` is useful for contract parsing and underlying close checks, but
  it does not contain bid, ask, open interest, IV, or Greeks.
- Massive option second aggregates can support a V1.5 trade-price proxy panel:
  for each candidate contract, the pipeline takes the latest VWAP or close from
  a pre-entry cutoff window and recomputes local IV. This route is explicitly
  labeled `no_nbbo_trade_proxy`; it can test event alignment and edge ranking,
  but it is not a full-spread executable backtest.
- `options_quotes_v1` is therefore required for quote-level work. Because a
  single daily quote file can be very large, the first paper run uses top 50
  names rather than top 150.
- Contract discovery is an explicit gate before quote extraction. The main
  sample keeps only standard equity options with `option_multiplier == 100`,
  `contract_size == 100`, and `deliverable_status == standard`. Non-standard OCC
  deliverables are reported as `non_standard_excluded` and never enter the quote
  pool.

Earnings events:

- SEC EDGAR company submissions are the primary historical event-candidate
  source. The pipeline uses 8-K / 8-K/A filings with Item 2.02 and keeps the SEC
  accession number and acceptance timestamp.
- Massive 8-K text is the validation layer. It is used to check whether Item
  2.02 text looks like a quarterly earnings release rather than another type of
  operational update.
- Nasdaq earnings calendar rows are auxiliary metadata only. In the first probe,
  historical timing was often `time-not-supplied`, including for large,
  SEC-confirmed earnings events.

SEC acceptance time is a regulatory timestamp. It is not guaranteed to be the
company's first public release time. For the first audited candidate route, it
is used only with a visible source flag and text-validation status.

## Models

The model order is deliberately conservative:

- Market-implied event variance.
- Last-four realized earnings variance.
- Last-four implied event variance.
- Goyal-Saretto-style RV-IV spread feature/baseline.
- Patell-Wolfson-style diagnostic features, based on pre-event
  implied-volatility behavior, realized earnings move history, and post-event
  volatility compression diagnostics.
- Linear or elastic-net model.
- LightGBM or XGBoost.
- FT-Transformer for tabular event features.
- Mamba sequence encoder for the 20-day pre-event option-surface path.

Deep learning is not assumed to win. If a tree model wins, the interpretation is
that event-level nonlinear tabular effects are sufficient for this task. If the
market benchmark wins after costs, the study still gives useful evidence about
the efficiency of earnings option prices under realistic trading frictions.

## Metrics

The evaluation has four layers.

Data integrity:

- Required field coverage.
- Quote-source and timestamp audit.
- IVAR extraction failure reasons.
- Exclusion counts by year, ticker, liquidity, VIX regime, and timing.

Forecast quality:

- MAE.
- RMSE.
- QLIKE.
- Out-of-sample `R^2` versus market-implied event variance.

Signal quality:

- AUC for cheap-versus-rich event classification.
- Brier score.
- Calibration by confidence bucket.
- Top-decile precision and hit rate.

Economic performance:

- Net PnL after transaction costs.
- Event-date aggregated Sharpe and Sortino.
- Max drawdown.
- Tail loss.
- Turnover and average cost.
- PnL by ticker, sector, VIX regime, liquidity, and BMO/AMC timing.

## Current Implementation State

Implemented:

- Strict local workflow through `just check`.
- 93% minimum coverage gate; current local gate is 51 tests with 94.23% total
  coverage.
- Event-window alignment for BMO/AMC.
- Timezone-aware event-entry timestamps.
- IVAR extraction with explicit failure reasons.
- Data, leakage, and fixture audit paths.
- Premium-space backtest smoke path.
- Model registry with honest implementation flags.
- Massive flat-file metadata and small day-aggregate probe.
- SEC-first earnings-candidate builder with optional Massive 8-K text validation.
- Contract discovery with standard-contract filtering before quote pooling.
- Event-panel diagnostics for forward source, ATM selection method, American
  option forward caveat, and possible preannouncement/prior-guidance review.
- V1.5 Massive second-aggregate trade-proxy panel, marked
  `no_nbbo_trade_proxy`, with pre-cutoff VWAP/close selection, local IV
  recomputation, gross proxy PnL, and haircut proxy PnL diagnostics.

Recent real sample:

- Command: `build-earnings-calendar` for AAPL, MSFT, and TSLA from 2026-01-01 to
  2026-04-30.
- Output: `artifacts/earnings_calendar_sample/`.
- Result: 8 SEC Item 2.02 candidates, 5 main-sample candidates after timing and
  text validation.
- Two TSLA Item 2.02 rows were classified as ambiguous operational updates, not
  clean quarterly earnings releases.
- One AAPL row was present in SEC but missing from the Massive text response at
  probe time.

Not implemented yet:

- Full top-50 event panel.
- Narrow extraction from daily `options_quotes_v1`.
- Production IV solver or vendor IV/Greeks/OI route.
- Full feature matrix.
- Train/test splits and model training.
- Paper-grade tables, figures, inference, and trading results.

## Interpretation

The empirical claim is deliberately narrow:

> Earnings option prices contain a market-implied event-variance benchmark. We
> test whether pre-event option-surface state dynamics and firm/event features
> add tradable information beyond that benchmark.

The central result is whether a model improves top-decile trade selection and
net PnL after costs. Forecast RMSE is reported, but it is not the main criterion.
The primary benchmark is market-implied event variance; model complexity matters
only if it improves the trading decision.

## Next Gate

The next implementation gate is a small but real panel:

1. Build the top-50 liquid option-stock universe for a recent test window.
2. Build an SEC-first earnings calendar for that universe, with Massive 8-K text
   validation and BMO/AMC/DMH/UNKNOWN flags.
3. Run the `trade-proxy-panel` stage on Massive option second aggregates to
   produce a `no_nbbo_trade_proxy` event panel for signal screening.
4. Extract the option quotes needed for event-expiry IVAR from `options_quotes_v1`
   without loading full daily files into memory.
5. Add local IV calculation or a documented vendor IV/Greeks/OI route.
6. Produce the first event-level table with RVAR, IVAR, extraction status,
   timing, liquidity, and text-validation status.

Use `just data` as the single data-engineering front door. Its default stage is
now the V1.5 Massive second-aggregate trade-proxy panel with `--max-events 10`,
`--jobs 4`, `--lookback-seconds 900`, and `--price-field option_vwap`. It writes
a manifest for each run, skips completed outputs by default, accepts `--force`
through `args` for overwrite, and can run multiple Massive metadata probe dates
in parallel with `--jobs`.

The data layout is a local lake:

- `data/bronze/`: source-preserving Massive flat files converted from temporary
  `.csv.gz` downloads into date-partitioned Parquet.
- `data/silver/`: validated calendar rows, event windows, underlying event
  bars, contract candidates, local IV estimates, and IVAR extraction inputs.
- `data/gold/`: analysis-ready event panels and, later, feature/model inputs.
- `artifacts/`: manifests, readiness reports, and audit summaries.

Polars is the default engine for large table reads and writes. The current
pilot-panel route uses `options_day_aggs` close prices as a provisional option
price proxy because entitled, narrow NBBO extraction from `options_quotes_v1`
is still the main performance-critical missing piece. The V1.5
`trade-proxy-panel` route improves the provisional layer by using pre-cutoff
Massive second-aggregate trade prices, while still preserving the no-NBBO
limitation in every output.
