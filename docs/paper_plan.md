# Research Design

Working title:

**Can Deep Learning Improve Earnings Volatility Trading? Evidence from U.S. Equity Options**

Technical title:

**State-Selective Event Variance Forecasting for Earnings Options: A Mamba-Based Approach with Risk-Defined Backtests**

## Core Question

The paper asks whether machine learning can improve the decision to trade
option-implied earnings event variance mispricing.

Primary forecast target:

```text
RVAR_event = log(S_after / S_before)^2
```

The ex post mispricing label is:

```text
RVAR_event - IVAR_event
```

The trading signal is premium-space:

```text
expected_strategy_edge_usd
  = expected_strategy_value_usd - market_entry_cost_usd
```

The primary entry rule is:

```text
expected_strategy_edge_usd > 1.5 * estimated_transaction_cost_usd
```

## Contribution

The contribution is not "Mamba predicts IV better." The intended contribution is:

> State-dependent pre-earnings option-surface dynamics may contain incremental
> information about event variance mispricing beyond market-implied event
> variance, historical earnings moves, and GBDT tabular baselines.

The interpretation depends on the empirical outcome:

- Mamba wins: pre-event sequence dynamics matter.
- LightGBM wins: event-level nonlinear tabular interactions are enough.
- Market-implied event variance wins after costs: the evidence is consistent
  with a hard-to-beat earnings option market under realistic frictions.
- Heterogeneity across liquidity, VIX, and BMO/AMC is an empirical question, not
  a directional claim.

## Related Literature and Positioning

The project sits at the intersection of earnings-announcement option pricing,
event-volatility trading, option-return predictability, and modern ML/DL model
comparison. The closest prior work is summarized below.

| Study | Full title and venue | Data and preprocessing | Methods / metrics | Main findings | Implication for this project |
| --- | --- | --- | --- | --- | --- |
| [Patell & Wolfson (1981)](https://ideas.repec.org/a/bla/joares/v19y1981i2p434-458.html) | "The Ex Ante and Ex Post Price Effects of Quarterly Earnings Announcements Reflected in Option and Stock Prices." *Journal of Accounting Research*, 19(2), 434-458. DOI: 10.2307/2490874. | Early listed-option and quarterly earnings-event data; event-study design around option and stock prices before and after announcements. | Ex ante and ex post price effects; implied-volatility behavior around the announcement window. | Establishes the classic fact pattern that uncertainty and implied volatility rise before earnings and resolve after the announcement. | Provides the baseline event-risk fact pattern, but predates modern option data, ML methods, and realistic transaction-cost backtests. |
| [Dubinsky, Johannes, Kaeck & Seeger (2019)](https://research.vu.nl/en/publications/option-pricing-of-earnings-announcement-risks/) | "Option Pricing of Earnings Announcement Risks." *The Review of Financial Studies*, 32(2), 646-687. DOI: 10.1093/rfs/hhy060. | Option prices are used to extract earnings-announcement uncertainty; this is the published version of the Dubinsky-Johannes earnings-announcement option-pricing line. | Reduced-form no-arbitrage earnings-jump models; Black-Scholes and stochastic-volatility extensions; option pricing-error comparisons. | Adding scheduled earnings jumps substantially reduces option pricing errors; earnings-announcement uncertainty has first-order pricing effects. | Directly supports modeling event variance separately instead of using `IV^2*T` as the target. |
| [Barth & So (2014)](https://www.gsb.stanford.edu/faculty-research/publications/non-diversifiable-volatility-risk-risk-premiums-earnings) | "Non-Diversifiable Volatility Risk and Risk Premiums at Earnings Announcements." *The Accounting Review*, 89(5), 1579-1607. | 1996-2007; more than 45,000 quarterly earnings announcements with traded options. | Extract implied announcement volatility, compare it with realized announcement volatility, and construct excess implied volatility. | Option prices embed an earnings-announcement risk premium related to non-diversifiable earnings risk. | Closest to our economic object, `IVAR_event - RVAR_event`; their focus is risk-premium/accounting asset pricing rather than ML ranking or tradable strategy selection. |
| [Donders, Kouwenberg & Vorst (2000)](https://ideas.repec.org/a/bla/eufman/v6y2000i2p149-171.html) | "Options and Earnings Announcements: An Empirical Study of Volatility, Trading Volume, Open Interest and Liquidity." *European Financial Management*, 6(2), 149-171. DOI: 10.1111/1468-036X.00118. | Option volume, open interest, and spreads around earnings announcements. | Event-time liquidity, volume, open-interest, and spread analysis. | Option volume and open interest rise before earnings; effective spreads are higher on the announcement day and next day. | Supports explicit liquidity filters, bid-ask cost modeling, and exclusion reporting. |
| [Gao, Xing & Zhang (2018)](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/anticipating-uncertainty-straddles-around-earnings-announcements/7B34877AD5E06304BA3C55FBA3219FDD) | "Anticipating Uncertainty: Straddles around Earnings Announcements." *Journal of Financial and Quantitative Analysis*, 53(6), 2587-2617. DOI: 10.1017/S0022109018000285. | OptionMetrics, 1996-2013; delta-neutral ATM straddles around earnings announcements. | ATM straddle holding windows such as `[-3,0]`; equal-, volume-, and open-interest-weighted portfolios; Newey-West t-statistics; Fama-MacBeth tests. | Individual straddles are usually negative-return trades, but pre-earnings straddles earn significantly positive returns; the `[-3,0]` window is around 3.34% in the published paper and is stronger among small, volatile, high-kurtosis, and less-liquid names. | Directly supports the tradability of earnings event volatility, while warning that apparent alpha may concentrate in noisy or illiquid stocks rather than mega-cap names. |
| [Chung & Louis (2017)](https://pure.psu.edu/en/publications/earnings-announcements-and-option-returns/) | "Earnings Announcements and Option Returns." *Journal of Empirical Finance*, 40, 220-235. DOI: 10.1016/j.jempfin.2016.07.010. | Before-earnings, after-earnings, and non-earnings straddle portfolios. | Sorts by earnings timing and prior volatility; reports straddle returns before and after transaction costs. | Before-earnings straddles earn positive returns, after-earnings straddles are more negative, and a two-way strategy earns 23.4% before bid/ask costs and 9.4% after bid/ask costs. | Supports the before-event long-volatility test and suggests post-announcement IV overreaction or recency bias as a useful contrast. |
| [Alexiou, Goyal, Kostakis & Rompolis (2025)](https://revfin.org/pricing-event-risk-evidence-from-concave-implied-volatility-curves/) | "Pricing Event Risk: Evidence from Concave Implied Volatility Curves." *Review of Finance*, 29(4), 963-1007. DOI: 10.1093/rof/rfaf016. | Short-term equity options around earnings-announcement dates. | Constructs short-expiry IV-curve concavity; evaluates earnings-day absolute abnormal returns, post-earnings realized volatility, and delta-neutral straddle/strangle/calendar returns. | Concave IV curves before earnings signal higher event risk and higher post-earnings realized volatility; option strategy returns are lower in concavity samples, consistent with investors paying a premium for gamma/event risk. | Provides the strongest support for `iv_butterfly_25d` / concavity as an incremental event-risk feature. |
| [Goyal & Saretto (2009)](https://doi.org/10.1016/j.jfineco.2009.01.001) | "Cross-Section of Option Returns and Volatility." *Journal of Financial Economics*, 94(2), 310-326. DOI: 10.1016/j.jfineco.2009.01.001. | Cross-section of equity options. | Sorts straddle and delta-hedged option portfolios by historical-volatility versus implied-volatility spreads. | Historical-realized versus implied-volatility spreads predict option returns. | Not earnings-specific, but important as a classical option-mispricing baseline. |
| [Chen, Gan & Vasquez (2023)](https://www.sciencedirect.com/science/article/pii/S0378426622003351) | "Anticipating Jumps: Decomposition of Straddle Price." *Journal of Banking & Finance*, 149, Article 106755. DOI: 10.1016/j.jbankfin.2022.106755. | Option straddle price decomposition using OptionMetrics, CRSP, Compustat, and IBES. | Decomposes straddles into volatility-risk and jump-risk assets and constructs S-jump. | S-jump rises before earnings announcements and predicts earnings-induced jump size and probability. | Closely related to our event-variance / jump-component object, but it is a structural decomposition rather than an ML trading-decision test. |

The first table defines the financial object. A second literature line is needed
to position the machine-learning design. These papers do not replace the
earnings-option literature; they justify the model comparisons, the use of
surface/path features, and the requirement that predictive gains be evaluated
economically rather than only by statistical loss.

| Study | Full title and venue | Data and preprocessing | Methods / metrics | Main findings | Implication for this project |
| --- | --- | --- | --- | --- | --- |
| [Hutchinson, Lo & Poggio (1994)](https://web.mit.edu/Alo/www/Papers/hutchinson-etal-94.html) | "A Nonparametric Approach to Pricing and Hedging Derivative Securities via Learning Networks." *The Journal of Finance*, 49(3), 851-889. DOI: 10.1111/j.1540-6261.1994.tb00081.x. | Simulated Black-Scholes option prices and S&P 500 futures options from 1987-1992. | Learning networks for option pricing and hedging; comparisons with OLS, kernel regression, projection pursuit, and multilayer perceptrons; out-of-sample pricing and delta-hedging performance. | Learning networks can recover the Black-Scholes pricing relation from training data and can price and delta-hedge options out of sample. | Establishes the early precedent for neural networks in option pricing, but the object is pricing/hedging formulas rather than earnings-event mispricing or risk-defined trading signals. |
| [Horvath, Muguruza & Tomas (2021)](https://portal.fis.tum.de/en/publications/deep-learning-volatility-a-deep-neural-network-perspective-on-pri/) | "Deep Learning Volatility: A Deep Neural Network Perspective on Pricing and Calibration in (Rough) Volatility Models." *Quantitative Finance*, 21(1), 11-27. DOI: 10.1080/14697688.2020.1817974. | Simulated and historical derivative-pricing data; implied volatility and option-price surfaces represented as grid-like inputs. | Neural networks approximate pricing functions for stochastic and rough-volatility models; calibration speed and pricing/calibration accuracy are the main metrics. | Neural-network surrogates can calibrate full implied-volatility surfaces in milliseconds and reduce the computational bottleneck in volatility-model calibration. | Supports treating option surfaces as structured model inputs, but it is a pricing/calibration paper, not an event-specific return-predictability or trading-strategy paper. |
| [Gu, Kelly & Xiu (2020)](https://academic.oup.com/rfs/article/33/5/2223/5758276) | "Empirical Asset Pricing via Machine Learning." *The Review of Financial Studies*, 33(5), 2223-2273. DOI: 10.1093/rfs/hhaa009. | Large U.S. equity return panel with many firm characteristics and macro predictors. | Broad model comparison across linear methods, regularized models, trees, random forests, gradient boosting, and neural networks; out-of-sample `R2` and portfolio Sharpe ratios. | ML gains come from nonlinearities and interactions; trees and neural networks can improve economic performance, but low signal-to-noise finance settings require disciplined out-of-sample evaluation. | Provides the empirical-asset-pricing template: compare many model classes fairly, guard against overfitting, and evaluate economic value, not only forecast error. |
| [Borochin & Zhao (2025)](https://ideas.repec.org/a/eee/empfin/v82y2025ics0927539825000404.html) | "The Economic Value of Equity Implied Volatility Forecasting with Machine Learning." *Journal of Empirical Finance*, 82, Article 101618. DOI: 10.1016/j.jempfin.2025.101618. | U.S. equity option data with monthly ATM implied-volatility innovation targets and option/firm predictors. | Classical models versus ML models such as classification/regression trees and bagged trees; out-of-sample IV-innovation forecasts; delta-hedged option portfolio sorts and return regressions. | ML improves out-of-sample IV-innovation forecasting, and forecast-sorted delta-hedged option portfolios show economically meaningful return differences; the strongest value is in forecast extremes. | Closest ML/economic-value comparator, but it predicts generic IV innovations rather than earnings event variance extracted from scheduled-jump windows. |
| [Hoefler (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4869272) | "Volatility Surfaces and Expected Option Returns." SSRN working paper, first posted 2024 and last revised 2025. DOI: 10.2139/ssrn.4869272. | Individual equity option volatility surfaces with option characteristics and return data. | CNN extracts signals from the implied-volatility surface; evaluates out-of-sample delta-hedged option-return forecasts, long-short portfolios, transaction costs, and feature importance. | Deep surface features forecast delta-hedged option returns; long-short spreads survive transaction costs and are strongest where intermediary constraints and retail-demand pressure matter. | Supports the idea that option-surface shape contains tradable information. Our design narrows that idea to short-expiry earnings-event variance and tests whether the tradable tail survives event-specific costs. |
| [Gorishniy, Rubachev, Khrulkov & Babenko (2021)](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) | "Revisiting Deep Learning Models for Tabular Data." *Advances in Neural Information Processing Systems* 34, 18932-18943. | Diverse supervised tabular datasets with numerical and categorical features. | Compares tabular deep models, ResNet-like baselines, FT-Transformer, XGBoost, and CatBoost under shared tuning and evaluation protocols. | FT-Transformer is a strong tabular deep-learning baseline, but there is no universally superior winner between GBDT and deep models. | Justifies FT-Transformer as the tabular DL benchmark and makes LightGBM/XGBoost a required baseline rather than an optional comparison. |
| [Gu & Dao (2024)](https://openreview.net/forum?id=AL1fq05o7H) | "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *Conference on Language Modeling* (COLM) 2024; arXiv:2312.00752. | General sequence-modeling benchmarks, including long-context sequence tasks. | Selective state-space model with input-dependent parameters; compares sequence-modeling performance and computational scaling against Transformer-style alternatives. | Input-dependent selective state propagation allows the model to retain or forget information along a sequence with linear-time scaling. | Provides the architecture rationale for a pre-earnings path encoder. In this project, Mamba is justified only if 20-day option-surface paths add information beyond event-level tabular features and GBDT baselines. |

The positioning is:

> Unlike prior work that documents average earnings straddle returns or estimates
> announcement risk premia, this paper tests whether models improve the
> cross-sectional ranking of option-implied event variance mispricing in the
> tradable tail after realistic costs.

The ML/DL literature is used to justify model choice, not to define the finance
claim. FT-Transformer is a tabular deep-learning benchmark that must be compared
against GBDT methods. Mamba is a sequence encoder for the pre-event
option-surface state path. Neither architecture is the contribution unless it
improves the tradable ranking beyond market-implied event variance and classical
baselines.

## Data

First-version source plan:

- Massive for U.S. options and underlying equity prices.
- SEC EDGAR company submissions as the primary historical earnings-event
  candidate route: use 8-K / 8-K/A Item 2.02 filings, SEC acceptance timestamps,
  and explicit BMO, AMC, DMH, and unknown timing flags derived from
  America/New_York market hours.
- Massive 8-K text as the validation route for parsed Item 2.02 content and
  non-earnings Item 2.02 exclusions.
- Nasdaq earnings calendar only as auxiliary expected-calendar metadata; it is
  not the primary historical timing source unless a later audit shows stable
  historical BMO/AMC coverage.
- Market controls such as SPY, sector ETFs, VIX, rates, dividends, and corporate
  actions where available.

The current code can already produce an auditable SEC-first candidate table with
Massive 8-K text validation. That table is a source-quality gate, not yet the
final paper event panel.

Target sample:

- Top 50 liquid U.S. single-name option stocks for the first paper run.
- 2013-2025.
- BMO and AMC only.
- Event-expiry DTE 5-14 in the main sample.
- Robustness DTE 3-21.
- Drop DMH and unknown announcement-time events in v1.

Top 50 is the main-sample default because the first empirical gate should
prioritize clean option-price/event alignment, storage telemetry, and API
coverage over universe breadth. A top-150 expansion is deferred until the
top-50 proxy data lake and later paper-grade quote/IV route are stable.

V1.5 is the active no-quote data route: Massive option second aggregates provide
pre-cutoff trade-price OHLCV bars for entry pricing and diagnostics, while
exit-date option day aggregates provide same-contract exit closes when
available. For each candidate contract, use the latest pre-entry VWAP or close
inside the resolved market-close cutoff window, recompute local IV, and require
paired call/put prices before constructing the event-expiry ATM input. These
outputs are labeled `no_nbbo_trade_proxy`. They can test whether the event
alignment, IVAR extraction, and edge-ranking pipeline has signal, but they do
not support full-spread crossing claims because they lack bid/ask and NBBO.
`options_quotes_v1` is a future paper-grade route/readiness probe, not an input
to the current proxy pipeline.

## Event Alignment

Timestamp alignment is a hard gate.
All event-entry timestamps are interpreted as 16:00 America/New_York. Feature
as-of timestamps must be timezone-consistent with the event-entry timestamp;
naive/aware timestamp mixtures are audit failures.

AMC:

- Trade before regular-session close on announcement date `d`.
- `S_before = close_d`.
- EOD event move: `log(close_{d+1} / close_d)`.

BMO:

- Trade before regular-session close on `d-1`.
- `S_before = close_{d-1}`.
- EOD event move: `log(close_d / close_{d-1})`.

DMH and unknown:

- Drop from the first version.
- Do not infer or repair timestamps unless a later audit validates the source.

SEC acceptance time is a regulatory timestamp, not automatically the company's
first public release timestamp. It is acceptable for the first audited event
candidate route only when the source audit records the filing accession,
acceptance timestamp, Item 2.02 status, text-classification result, and
BMO/AMC/DMH/unknown flag. Any event whose release-time interpretation is
ambiguous stays outside the main sample.

Macro and market feature cutoffs:

- AMC events use only features available through regular close on date `d`.
- BMO events use only features available through regular close on `d-1`.
- Every feature row must carry `feature_asof_timestamp` and
  `event_entry_timestamp`.

## Target Variables

Main close-to-close realized event variance:

```text
RVAR_event = log(S_after / S_before)^2
```

where AMC uses `close_{d+1}/close_d` and BMO uses `close_d/close_{d-1}`.

Event variance mispricing:

```text
Mispricing = RVAR_event - IVAR_event
```

The headline target is the mispricing direction. Regression results are support
evidence; ranking and trading results are the main evidence.

## Implied Event Variance

Avoid using `IV^2 * T` as the event target because it mixes ordinary diffusive
variance with the scheduled earnings event.

Use total ATM implied variance:

```text
w(T) = sigma_ATM(T)^2 * T
```

For two expiries that both cover the earnings event, assume:

```text
w1 = d*T1 + v_e
w2 = d*T2 + v_e
```

Then extract:

```text
IVAR_event = (T2*w1 - T1*w2) / (T2 - T1)
```

`T_j` uses the same year-fraction convention as the IV source. V1 uses ACT/365
unless vendor documentation specifies otherwise. Mixed time conventions are
forbidden within one extraction run.

The two expiries must cover the realized event window, not merely the calendar
announcement date. When the event-window exit date is unavailable, v1
conservatively excludes same-date expiries.

ATM selection and forward convention:

- Primary ATM is ATMF when a liquid near-ATM call/put pair supports a
  put-call-parity implied forward.
- Because the sample is U.S. single-name American options, that forward is an
  approximation, not a theoretical equality. V1 treats the early-exercise
  premium as negligible only for short-DTE, no-dividend, near-ATM contracts.
- If the parity pair is missing, stale, wide, outside DTE 5-14, or the event
  window through the second IVAR expiry crosses an ex-dividend date, fall back
  to nearest-spot ATM and record `forward_source = spot_fallback`.
- The event panel records `forward_source`, `forward_price`,
  `atm_selection_method`, and `american_forward_caveat_flag`.

If extracted `IVAR_event < 0`, flag the event as a term-structure extraction
failure. The conservative v1 behavior is to exclude it from the tradable sample
and report the failure rate.

Nonmonotone total variance is also excluded and reported. No smoothing,
monotonicity projection, or surface repair is applied in v1.

## Filters

Main mechanical filters:

```text
S_t > 5
bid > 0
ask > bid
(ask - bid) / mid < 0.30
OI > 0 or volume > 0
5 <= event_expiry_DTE <= 14
option_multiplier == 100
contract_size == 100
deliverable_status == standard
```

The robustness sample expands DTE to `3 <= DTE <= 21`.

Non-standard OCC contracts are excluded before proxy-price or paper-grade quote
pooling. Splits, spinoffs, special dividends, or reference rows with non-100
deliverables get
`contract_discovery_status = non_standard_excluded`. They are reported in the
discovery table but never enter IV or backtest inputs.

## Features

V1 event-level feature families:

- `IVAR_event`.
- ATM IV for event expiry and next expiry.
- Term spread.
- RR25 and BF25.
- Short-expiry `iv_butterfly_25d`.
- Paper-grade only: bid-ask spread and quote liquidity.
- V1.5 only: second-aggregate trade-price proxy activity, including last
  pre-cutoff trade time, window volume, transaction count, and stale-window
  status. This is a screening feature family, not a substitute for bid-ask
  costs.
- Option volume and open interest.
- RV5, RV20, RV60.
- Prior earnings move and last-four earnings average move.
- SPY return, sector ETF return, VIX.
- BMO/AMC dummy.
- Market-cap or liquidity bucket.

Butterfly feature:

```text
iv_butterfly_25d = IV_25P - 2*IV_ATM + IV_25C
```

This is a key differentiator from plain ATM-IV and skew designs because
short-expiry IV curve shape can proxy earnings-day bimodality and event-risk
pricing.

## Model Ladder

The model order should start from financial benchmarks and then add model
complexity only when it can be evaluated against those benchmarks.

Required baselines:

1. Market-implied baseline: `RVAR_hat = IVAR_event`.
2. Last-four RVAR baseline: same-ticker average of the prior four realized
   earnings event variances.
3. Last-four IVAR baseline.
4. Goyal-Saretto-style RV-IV spread feature/baseline, not a full replication of
   the original portfolio design.
5. Patell-Wolfson-style diagnostic features, not a trainable model. Inputs are
   pre-event implied-volatility behavior, realized earnings move history, and
   post-event volatility compression diagnostics.
6. Linear or elastic-net event model.
7. LightGBM or XGBoost.

Deep models:

1. FT-Transformer for event-level tabular features.
2. Mamba sequence encoder for the pre-event 20-day state path.

Mamba input:

```text
X_i = {x_{t-20}, ..., x_{t0}}
```

Each daily state should include ATM IV, event IVAR estimate, skew, butterfly,
concavity, term spread, spread, volume, RV5, SPY return, and VIX.

## Losses

Use quantile loss as the main training loss because earnings moves are heavy
tailed:

```text
rho_tau(u) = u * (tau - 1{u < 0})
```

Trading signals should use median forecasts plus uncertainty filters.

Example long-vol rule:

```text
Q50(RVAR_event) > IVAR_event
Q10(RVAR_event) > IVAR_event - buffer
```

Example short-vol rule:

```text
Q50(RVAR_event) < IVAR_event
Q90(RVAR_event) < IVAR_event + buffer
```

## Backtests

Primary strategies:

1. Long ATM straddle for predicted cheap event volatility.
2. Short iron fly for predicted rich event volatility.

Calendar straddles are a second-stage relative-value extension, not the first
headline strategy, because they mix event variance, post-event vega, theta,
front/back gamma mismatch, skew, dividends, borrow, and assignment risk.

Premium-space valuation:

- V1 primary uses deterministic quadrature under a zero-mean Gaussian event
  return with variance `forecast_RVAR_event`.
- The deterministic smoke backtest still uses a simplified payoff layer. The
  Massive V1.5 trade-proxy diagnostics mark exit using the same contracts'
  exit-date option day-aggregate closes when available, preserving residual
  extrinsic value; intrinsic payoff is only a flagged fallback for missing exit
  option closes or 0DTE expiry.
- Robustness hooks include symmetric two-point, Student-t, empirical residual,
  and later mixed-normal/Laplace jump distributions.
- Multi-leg fills assume simultaneous quoted bid/ask execution; legging risk is
  documented but not modeled in v1.

Transaction-cost reporting:

- Mid price: theoretical upper bound.
- Half-spread: optimistic institutional execution.
- Full bid-ask crossing: conservative tradable result.

The paper-facing result must emphasize full-spread crossing.

## Splits

Do not randomly split events.

Default split:

```text
Train: 2013-2018
Validation: 2019-2020
Test: 2021-2025
```

Preferred final design:

- Rolling walk-forward.
- Train five years.
- Validate one year.
- Test the next year.
- Purge adjacent same-ticker earnings leakage.
- Do not randomly split same-date peer-firm events.

## Diagnostics

Report exclusion counts and shares for DMH/unknown timing, DTE, liquidity,
ex-dividend, halts, missing fields, insufficient sequence history, missing
ATM/wing legs, and IVAR failures. Cross-tab exclusions by year and VIX regime.

Negative and nonmonotone IVAR diagnostics are outputs, not v1 model inputs.
`negative_ivar_by_term_structure_bucket.csv` should include ticker, event date,
selected expiries, selected DTEs, `expiry_gap_days`, moneyness, spread width,
`iv_used_for_extraction_1`, `iv_used_for_extraction_2`, total variances, failure
reason, and VIX regime. These raw inputs are kept even when IVAR extraction
fails.

Pre-announcements are a known residual risk. V1 does not run NLP to detect prior
guidance automatically. Instead, error analysis flags events with high
`IVAR_event`, very low `RVAR_event`, and a large negative realized
`RVAR_event - IVAR_event` as `possible_preannouncement_or_prior_guidance` for
manual review. This is a diagnostic tag, not an automatic exclusion.

## Evaluation

Forecast layer:

- MAE.
- RMSE.
- QLIKE.
- Out-of-sample `R2` against the market-implied baseline.

Mispricing-classification layer:

- AUC.
- Brier score.
- Calibration curve.
- Precision at top decile.
- Hit rate by confidence bucket.

Strategy layer:

- Net return.
- Sharpe and Sortino.
- Max drawdown.
- Hit rate.
- Average win and loss.
- Tail loss.
- Turnover.
- Average bid-ask cost.
- PnL by ticker, sector, VIX regime, liquidity bucket, and BMO/AMC.

Inference layer:

- Event-date clustered standard errors.
- Ticker clustered standard errors.
- Two-way clustered standard errors.
- Block bootstrap confidence intervals.
- SPA or model-confidence-set tests if many models and thresholds are compared.

## Paper Structure

1. Introduction.
2. Literature on earnings option pricing, event volatility, and ML/DL forecasting.
3. Data and variable construction.
4. Models.
5. Forecast and mispricing-classification results.
6. Risk-defined backtests.
7. Robustness.
8. Conclusion.

The conclusion should stay disciplined:

> Deep learning is useful only if it improves the ranking of event variance
> mispricing in the tradable tail of the distribution.
