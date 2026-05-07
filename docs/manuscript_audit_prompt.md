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
- that Mamba is the contribution independent of baselines, ablations, and
  costs;
- that calendar spreads isolate pure event variance.

## Required Manuscript Elements

Research object:

- Earnings announcements are scheduled jump-risk events.
- Options embed an ex ante market-implied event variance.
- Models forecast C2O, C2C, and O2C realized-variance targets.
- The primary scientific target is `RVAR_event_jump_c2o`.
- V1 tradable proxy mispricing uses C2C: `RVAR_event_day_c2c - IVAR_event`.
- Trading entry is evaluated in USD premium space, not raw variance space.

Data:

- The options source and entitlement window are stated explicitly.
- Current local proxy results use Massive option second aggregates and option
  day aggregates from the observed 2022-onward entitlement window.
- Market-data inputs are stated precisely: options day aggregates for universe,
  contract, IV proxy, exit, and sequence construction; underlying day
  aggregates for closes and `RVAR_event`; targeted option one-second trade
  aggregates for entry proxy pricing.
- Entry second aggregates are restricted to the pre-cutoff buffer, default 60
  minutes before event cutoff, and the selected entry price comes from the
  latest positive VWAP or close in the final 900 seconds before cutoff.
- The 2013-2025 sample is described as the target paper range unless historical
  option data for that range has actually been acquired and processed.
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
- FT-Transformer and Mamba are positioned after strong tabular baselines.
- Mamba results include sequence coverage, drop-rate diagnostics, and a
  mask-only ablation.
- If LightGBM/XGBoost beat Mamba, the conclusion is that tabular nonlinear
  interactions currently dominate the proxy sequence route.

Evaluation:

- Forecast metrics include MAE, RMSE, QLIKE, and OOS R2 versus market-implied
  IVAR.
- Ranking metrics include AUC, Brier score, calibration, and top-decile
  precision.
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

- "Mamba predicts IV better" as the headline.
- Random train/test split.
- Missing BMO/AMC alignment.
- Trades entered after the event cutoff.
- Second-aggregate trade bars described as quotes, mid, bid/ask, or NBBO.
- Full-spread results omitted while making executable strategy claims.
- Deep models compared only against weak neural baselines and not LightGBM or
  XGBoost.
- Calendar returns interpreted as pure event-variance returns.
- Variance-space edge compared directly to dollar transaction costs.
- Proxy results from 2022-2025 presented as full 2013-2025 paper evidence.
