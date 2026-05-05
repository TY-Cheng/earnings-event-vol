# Manuscript Audit Brief

Audit any manuscript for consistency with the `earnings-event-vol` research
contract.

## Paper Claim

The manuscript should claim:

> We test whether machine learning improves the ranking of option-implied
> earnings event variance mispricing in tradable, risk-defined option strategies.

It should not claim:

- generic IV forecasting superiority;
- that lower RMSE alone implies economic value;
- that Mamba is the contribution independent of baselines and costs;
- that calendar spreads isolate pure event variance.

## Required Manuscript Elements

Introduction:

- Earnings announcements are scheduled jump-risk events.
- Options embed an ex ante market-implied event variance.
- The paper asks whether models improve the forecast and ranking of realized
  event variance relative to that market-implied benchmark.

Data:

- Massive or another named options source is described clearly.
- Earnings timestamp source and BMO/AMC rules are explicit.
- DMH and unknown timing are excluded in v1.
- Bid/ask, liquidity, DTE, and event-alignment filters are reported.
- Any second-aggregate or trade-price proxy result is labeled as
  `no_nbbo_trade_proxy` and separated from bid/ask executable backtests.

Variable construction:

- `RVAR_event = log(S_after / S_before)^2`.
- `IVAR_event` is extracted from two-expiry total implied variance.
- Entry thresholds are expressed in USD premium space, not variance space.
- Negative event-variance extractions are flagged, excluded, and reported.
- `iv_butterfly_25d` is defined from short-expiry IV curve shape.

Models:

- Market-implied event variance is the primary baseline.
- Historical earnings moves, linear or elastic-net, and LightGBM are included.
- FT-Transformer and Mamba are positioned after strong baselines.
- Quantile loss is justified by heavy-tailed earnings moves.

Backtests:

- Long ATM straddle tests predicted cheap event volatility.
- Short iron fly tests predicted rich event volatility.
- Full bid-ask crossing is the main cost assumption.
- Mid and half-spread results are labeled as sensitivity cases.
- Multi-leg fills assume simultaneous quoted bid/ask execution and disclose
  legging risk as an unmodeled limitation.

Inference:

- Event-date and ticker clustering are reported.
- Cross-sectional dependence is not evaluated with naive t-stats only.
- Threshold/model multiple testing is handled if many variants are compared.

## Red Flags

- "Mamba predicts IV better" as the headline contribution.
- Random train/test split.
- Missing BMO/AMC alignment.
- Trades entered after the earnings announcement quote date.
- Full-spread results omitted.
- Second-aggregate trade bars presented as if they were NBBO quotes.
- Deep models compared only against LSTM or MLP and not LightGBM.
- Calendar returns interpreted as pure event-variance returns.
- Variance-space edge compared directly to dollar transaction costs.
