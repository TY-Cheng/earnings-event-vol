# Manuscript Audit Brief

Audit any manuscript or slide narrative against the `earnings-event-vol`
research contract.

## Valid Claim Shape

The manuscript may claim, if supported by the reported evidence:

> We test whether models improve the ranking of option-implied earnings event
> variance mispricing in risk-defined earnings option strategies.

For the current local results, the claim must stay narrower:

> In a no-NBBO proxy sample, model features show preliminary cross-sectional
> ranking signal for earnings event-variance mispricing. Paper-grade tradability
> requires bid/ask or NBBO execution data.

It must not claim:

- generic IV forecasting superiority;
- paper-grade executable backtest results from second-aggregate trade bars;
- that lower RMSE alone implies economic value;
- that a sequence model is the contribution
  independent of baselines, ablations, and costs;
- that calendar spreads isolate pure event variance.

## Required Manuscript Elements

Research object:

- Earnings announcements are scheduled jump-risk events.
- Options embed an ex ante market-implied event variance.
- Models forecast C2O, C2C, and O2C realized-variance targets.
- The primary scientific target is `RVAR_event_jump_c2o`.
- Tradable proxy mispricing uses C2C: `RVAR_event_day_c2c - IVAR_event`.
- Trading entry is evaluated in USD premium space, not raw variance space.

Data:

- The options source and entitlement window are stated explicitly.
- Current main no-NBBO data, feature, model, and report artifacts target
  `2016-10-01` through `2026-06-05` in the WSL2/CUDA cold-run root
  `/home/tycheng/data/earnings-event-vol`. Older repo-local
  artifacts, older bounded quote slices, and broader preflight materializations
  must be labeled stale or preflight only.
- Market-data inputs are stated precisely: options day aggregates for universe,
  contract, IV proxy, fallback exit diagnostics, and sequence construction;
  underlying day aggregates for vendor OHLC opens, C2O/C2C/O2C targets, and
  exit spot; targeted option one-second trade aggregates for entry pricing,
  primary C2C exit pricing, and post-open C2O/O2C open-anchor pricing.
- Entry second aggregates are restricted to the pre-cutoff buffer, default 60
  minutes before event cutoff, and the selected entry price is the true per-leg
  volume-weighted option VWAP over the final 900 seconds before cutoff.
- The option open anchor is unified as a trade-aggregate 5-15 minute post-open
  VWAP. It is the primary C2O comparison mark and the O2C diagnostic entry
  mark; 0-5 minute VWAP is only an opening microstructure stress test.
- C2C exits use exit-date preclose 15-minute option VWAP as the primary proxy;
  option day-aggregate close is not used as a strategy-exit fallback.
- O2C proxy PnL is a realized decomposition diagnostic, not a model-driven
  strategy headline without a post-open residual-IV baseline.
- The 2016-10-01 to 2026-06-05 no-NBBO cold-run sample may be described as
  processed for proxy-stage research; paper-grade execution still requires
  historical bid/ask or NBBO-equivalent data for that range.
- Earnings events come from SEC EDGAR submissions plus SEC primary filing
  document validation.
- Massive 8-K text is auxiliary fallback only.
- BMO/AMC rules are explicit; DMH and unknown events are excluded in v1.
- Universe construction filters non-single-name symbols before selecting the
  monthly top 50.
- Any second-aggregate or trade-price proxy result is labeled
  `no_nbbo_trade_proxy` and separated from bid/ask executable backtests.

Variables:

- `RVAR_event_jump_c2o = log(open_after / close_before)^2`.
- `RVAR_event_day_c2c = log(close_after / close_before)^2`.
- `rvar_event` is the C2C backward-compatible alias.
- `IVAR_event` is extracted from two-expiry total implied variance.
- IVAR extraction failures are reported by reason, including missing event-
  covering expiries, nonmonotone total variance, and negative extracted IVAR.
- Feature as-of timestamps are before or at event entry.
- `iv_butterfly_25d` or proxy curve-shape measures are defined before use.

Models:

- Market-implied IVAR is the primary benchmark.
- Historical event baselines and Goyal-Saretto-style RV-IV spread are included.
- Elastic Net and LightGBM/XGBoost are included before deep-model claims.
- FT-Transformer and sequence diagnostics are positioned after strong tabular
  baselines.
- Sequence results include coverage, drop-rate diagnostics, mask-only controls,
  and deterministic time-shuffle controls.
- If LightGBM/XGBoost beat the sequence suite, the conclusion is that tabular
  nonlinear interactions currently dominate the proxy sequence route.

Evaluation:

- Forecast metrics include MAE, RMSE, QLIKE, and OOS R2 versus market-implied
  IVAR.
- Ranking metrics include AUC, rank-probability Brier diagnostics, calibration
  diagnostics, and top-decile precision.
- Strategy metrics include net proxy PnL, return on premium or capital, Sharpe,
  Sortino, drawdown, hit rate, tail loss, turnover, and cost sensitivity.
- Inference does not rely on naive t-stats only; event-date, ticker, two-way
  clustering, block bootstrap, or model-comparison corrections are used when
  the paper moves beyond proxy screening.

Backtests:

- Long ATM straddle tests predicted cheap event volatility.
- Short iron fly tests predicted rich event volatility.
- Current proxy PnL is explicitly non-NBBO and non-paper-grade.
- Paper-grade execution claims require historical bid/ask or NBBO, realistic
  spread crossing, and leg-level cost accounting.
- Mid or haircut results are labeled as sensitivity or proxy cases, not the
  main tradability evidence.
- Multi-leg fills disclose simultaneous-fill assumptions and unmodeled legging
  risk.

## Red Flags

- "A sequence model predicts IV better" as the headline.
- Random train/test split.
- Missing BMO/AMC alignment.
- Trades entered after the event cutoff.
- Second-aggregate trade bars described as quotes, mid, bid/ask, or NBBO.
- Full-spread results omitted while making executable strategy claims.
- Deep models compared only against weak neural baselines and not LightGBM or
  XGBoost.
- Calendar returns interpreted as pure event-variance returns.
- Variance-space edge compared directly to dollar transaction costs.
- Historical proxy/model results presented as current target-window paper
  evidence without matching the current cold-run manifests.
