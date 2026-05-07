---
hide:
  - navigation
---

# Results Snapshot

This page is the curated reader-facing snapshot. Raw generated outputs live
under ignored `artifacts/`, `data/`, and `reports/` paths; only selected
results and figures are copied into `docs/` for publication.

## Status

Current local run state, verified on 2026-05-07:

| Item | Current state |
| --- | --- |
| Data route | SEC-first event calendar plus Massive market-data proxy |
| Execution grade | `no_nbbo_trade_proxy`, `paper_grade=false` |
| Study window in current run | 2022-12-01 to 2025-12-31 |
| Target paper window | 2013-2025, pending historical quote/NBBO or equivalent data |
| Universe | Monthly top 50 liquid U.S. single-name option underlyings |
| Market data route | Options day aggregates, underlying day aggregates, and targeted option one-second trade aggregates |
| Event source | SEC EDGAR submissions plus SEC primary filing document text |
| Main timing sample | BMO and AMC only |
| Research package | Target-aware C2O/C2C/O2C feature matrix, model metrics, proxy strategy diagnostics, figures |

The current evidence is useful for engineering validation and signal screening.
It is not paper-grade execution evidence because it does not use historical
bid/ask or NBBO quotes.

## Data and Backtest Setup

The current proxy route uses three required market-data inputs:

- **Options day aggregates** for dynamic-universe liquidity ranking, contract
  discovery, local IV/IVAR proxy inputs, fallback exit diagnostics, and the
  daily close-trade-implied option-surface sequence.
- **Underlying stock day aggregates** for underlying closes, vendor OHLC opens,
  C2O/C2C/O2C event returns, and exit spot.
- **Option one-second trade aggregates** for targeted pre-cutoff entry proxies,
  primary C2C exit proxies, and post-open option open-anchor proxies. Massive REST is queried with
  `/range/1/second/<date>/<date>`. The entry cache keeps only bars in the
  resolved pre-cutoff buffer, default 60 minutes before the event cutoff.
  Entry pricing uses the true per-leg volume-weighted `option_vwap` over the
  final 900 seconds before cutoff. The unified option open anchor is
  same-contract option VWAP from 5-15 minutes after open: it is the primary C2O
  exit proxy and the O2C diagnostic entry proxy. The 0-5 minute VWAP remains an
  opening-microstructure stress test. C2C exits use same-contract option VWAP
  over the final 15 minutes before the exit-date close; option day-aggregate
  close is fallback/diagnostic only. The same pre-cutoff cached bars can feed the
  12-bin entry-day intraday proxy sequence.

Candidate contracts are validated against Massive option reference metadata
before entry proxy fetching. Non-100 or adjusted deliverables are marked
`non_standard_excluded`; reference-fetch failures are diagnostics and do not
create paper-grade execution claims.

The one-second aggregates are trade OHLCV bars. They are not quote midpoints,
bid/ask records, or NBBO. Intraday sequence features are therefore
trade-aggregate proxy surfaces, not observed NBBO-mid IV surfaces.

Market-state controls are availability-gated with `just data
market-second-covariates`: SPY/QQQ option one-second aggregates and SPY/QQQ
underlying one-second aggregates at the event entry cutoff. When present, those
controls add entry-as-of ATM IV proxy, term slope, skew, butterfly, straddle
premium over spot, option activity, and underlying pre-cutoff return. Missing
coverage is reported and the no-market-control specification remains valid.
They follow the same `no_nbbo_trade_proxy` caveat.

The underlying universe is dynamic rather than a fixed ticker list:

1. Start from SEC company ticker metadata and keep eligible U.S. common-equity
   tickers on supported exchanges.
2. Exclude ETF, fund, trust, ETN, index, volatility, commodity, and other
   non-single-name-like symbols before liquidity ranking.
3. Parse each option ticker to its underlying and compute monthly option
   premium dollar volume:

   ```text
   option_premium_dollar_volume = option_price * contract_volume * 100
   ```

   where `option_price` uses VWAP when available and close as the fallback.
4. For each universe month, rank underlyings by trailing six-month option
   premium dollar volume, excluding the current month to avoid look-ahead.
5. Keep the top 50 underlyings and pass their ticker union to the SEC-first
   earnings-calendar stage. Each event is then annotated with
   `universe_month`, `universe_rank`, and `universe_filter_status`.

This setup is part of both the data construction and the model/backtest design:
it defines the tradable sample, the entry/exit proxy, and which observations
enter the feature matrix and strategy diagnostics.

## Data Coverage

Latest proxy data pipeline outputs:

| Measure | Value |
| --- | ---: |
| Dynamic-calendar rows | 1,054 |
| BMO/AMC main-sample candidates | 810 |
| Trade-proxy event-panel rows | 810 |
| Events with C2C `rvar_event` alias | 801 |
| Events with trade-proxy `IVAR_event` | 693 |
| Proxy contract candidates | 12,038 |
| Contracts with usable pre-cutoff proxy price | 10,165 |
| Contracts with no trade in cutoff window | 1,873 |
| Main DTE 5-14 contracts | 5,098 |
| Robustness DTE 3-21 contracts | 12,038 |
| Proxy straddle diagnostic rows | 779 |

IVAR failure diagnostics:

| Failure reason | Events |
| --- | ---: |
| No two event-covering expiries | 103 |
| Nonmonotone total variance | 7 |
| Negative extracted IVAR | 7 |

Proxy straddle diagnostics:

| Measure | Value |
| --- | ---: |
| Mean gross C2C primary proxy PnL | 18.15 USD |
| Mean haircut C2C proxy PnL | -153.42 USD |
| Mean C2O intrinsic-open gross diagnostic PnL | -384.24 USD |
| Mean C2O post-open option VWAP 0-5m proxy PnL | -21.14 USD |
| Mean C2O post-open option VWAP 5-15m proxy PnL | -7.31 USD |
| Mean O2C option VWAP 5-15m to primary C2C exit diagnostic PnL | 8.65 USD |
| Mean option-proxy decomposition residual, 5-15m | 0.00 USD |

Interpretation: the proxy route produces a usable event panel, but the IVAR
coverage gap is still material. The main loss channel is missing a valid pair of
event-covering expiries, followed by smaller term-structure extraction failures.

## Model Results

The feature matrix has 810 rows. The current chronological proxy split trains on
567 rows, validates on 121 rows, and tests on 122 rows for tabular models. The
target system is:

- `jump_c2o`: primary scientific target, close-to-open earnings jump variance.
- `day_c2c`: literature-compatible target and the only V1 proxy-PnL headline.
- `reaction_o2c`: full regular-session open-to-close diagnostic for post-open
  digestion.

Daily Mamba uses a 20 x 37 tensor. Hybrid Mamba uses a 31 x 21 mixed-clock tensor
with 19 prior daily proxy-surface states plus 12 entry-day five-minute
trade-aggregate proxy bins. Sequence coverage is 678 eligible events out of 810,
so Mamba results are diagnostic rather than headline.

Selected `jump_c2o` forecast metrics:

| Model | N | MAE | RMSE | OOS R2 vs IVAR |
|:---|---:|---:|---:|---:|
| Market IVAR | 99 | 0.0097 | 0.0146 | 0.000 |
| Goyal-Saretto spread | 99 | 0.0079 | 0.0138 | 0.107 |
| Elastic Net | 122 | 0.0093 | 0.0211 | 0.334 |
| LightGBM | 122 | 0.0079 | 0.0193 | 0.416 |
| XGBoost | 122 | 0.0077 | 0.0190 | 0.409 |
| FT-Transformer | 122 | 0.0374 | 0.0395 | -5.357 |
| Daily Mamba 20-step | 100 | 0.0067 | 0.0136 | 0.121 |
| Hybrid Mamba 31-step | 100 | 0.0082 | 0.0228 | 0.192 |
| Intraday-only Mamba 12-step | 100 | 0.0083 | 0.0228 | 0.190 |
| Mask-only hybrid Mamba | 100 | 0.0088 | 0.0242 | -0.040 |

Selected `jump_c2o` ranking metrics:

| Model | N | Top-decile precision | AUC | Brier |
|:---|---:|---:|---:|---:|
| Market IVAR | 99 | 0.000 | 0.500 | 0.254 |
| Goyal-Saretto spread | 99 | 0.300 | 0.606 | 0.310 |
| Elastic Net | 99 | 0.600 | 0.594 | 0.313 |
| LightGBM | 99 | 0.500 | 0.697 | 0.286 |
| XGBoost | 99 | 0.300 | 0.733 | 0.277 |
| FT-Transformer | 99 | 0.100 | 0.496 | 0.338 |
| Daily Mamba 20-step | 86 | 0.222 | 0.487 | 0.341 |
| Hybrid Mamba 31-step | 92 | 0.200 | 0.452 | 0.349 |
| Intraday-only Mamba 12-step | 92 | 0.100 | 0.451 | 0.350 |
| Mask-only hybrid Mamba | 92 | 0.200 | 0.466 | 0.346 |

Selected `day_c2c` proxy strategy metrics:

| Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|---:|---:|---:|---:|---:|
| Last-four RVAR | 100 | 2,611.31 | 0.015 | 0.207 | -12,290.67 |
| Last-four IVAR | 100 | -8,783.43 | -0.052 | -0.697 | -9,976.33 |
| Goyal-Saretto spread | 100 | 3,429.89 | 0.020 | 0.272 | -16,624.92 |
| Elastic Net | 100 | 52,032.24 | 0.307 | 4.560 | -2,328.78 |
| LightGBM | 100 | 68,251.65 | 0.403 | 6.502 | -1,272.29 |
| XGBoost | 100 | 66,912.65 | 0.395 | 6.321 | -1,037.01 |
| FT-Transformer | 100 | 3,440.02 | 0.020 | 0.273 | -9,853.08 |
| Daily Mamba 20-step | 87 | -8,588.00 | -0.061 | -0.714 | -14,517.46 |
| Hybrid Mamba 31-step | 93 | -3,163.55 | -0.020 | -0.253 | -12,213.45 |
| Intraday-only Mamba 12-step | 93 | -3,163.55 | -0.020 | -0.253 | -12,213.45 |
| Mask-only hybrid Mamba | 93 | -3,027.79 | -0.019 | -0.242 | -12,213.45 |

Selected `jump_c2o` post-open option-VWAP proxy PnL:

The primary C2O option proxy exits the same selected straddle using
same-contract option VWAP from 5-15 minutes after the regular-session open.
The 0-5 minute VWAP is an opening-microstructure stress test, and the
intrinsic-open mark remains a pure underlying-jump diagnostic. All are
`no_nbbo_trade_proxy`, not NBBO-executable PnL.

The same 5-15 minute option VWAP is also the unified O2C open anchor. O2C PnL
is reported as a realized decomposition diagnostic from that open anchor to the
primary C2C exit mark, not as a model-driven strategy headline; a true O2C
strategy needs a post-open residual-IV baseline.

| Proxy | Model | Trades | Net proxy PnL | Return on premium | Sharpe | Max drawdown |
|:---|:---|---:|---:|---:|---:|---:|
| C2O VWAP 5-15m | Elastic Net | 93 | 14,324.21 | 0.085 | 1.335 | -5,844.42 |
| C2O VWAP 5-15m | LightGBM | 93 | 36,405.45 | 0.215 | 3.599 | -3,629.01 |
| C2O VWAP 5-15m | XGBoost | 93 | 38,878.75 | 0.230 | 3.884 | -1,941.38 |
| C2O VWAP 5-15m | Hybrid Mamba 31-step | 88 | -7,438.48 | -0.047 | -0.689 | -10,292.62 |
| C2O VWAP 0-5m | LightGBM | 95 | 32,297.13 | 0.191 | 3.341 | -4,423.65 |
| C2O VWAP 0-5m | XGBoost | 95 | 38,509.78 | 0.227 | 4.092 | -1,980.30 |
| C2O intrinsic-open | LightGBM | 100 | 59,591.91 | 0.352 | 5.052 | -3,681.98 |
| C2O intrinsic-open | XGBoost | 100 | 52,011.49 | 0.307 | 4.277 | -3,654.03 |

Interpretation: in this no-NBBO proxy run, the strongest evidence is the
`jump_c2o` ranking signal from tabular models and the `day_c2c` proxy economic
screen from XGBoost, LightGBM, and Elastic Net. The C2O 5-15m option-VWAP proxy
confirms the same tabular ranking story under a more realistic post-open trade
aggregate mark, while the intrinsic-open diagnostic is only a lower-bound jump
diagnostic. Mamba is implemented across daily, hybrid, intraday-only, and
mask-only variants, but it is not the headline model in the current run.

The market-implied IVAR baseline is still the central benchmark. It generates no
trades under the premium-edge rule because its forecast edge is zero by
construction.

## Figures

Forecast error:

![Forecast performance](assets/images/modeling/forecast_performance.png)

Ranking quality:

![AUC and top-decile precision](assets/images/modeling/auc_top_decile_precision.png)

Proxy strategy PnL:

![Strategy PnL by edge decile](assets/images/modeling/strategy_pnl_by_edge_decile.png)

Cost sensitivity:

![Cost sensitivity](assets/images/modeling/cost_sensitivity.png)

Calibration:

![Calibration plot](assets/images/modeling/calibration_plot.png)

Realized mispricing by predicted edge decile:

![Edge decile realized mispricing](assets/images/modeling/edge_decile_realized_mispricing.png)

QLIKE contribution diagnostic:

![QLIKE contribution diagnostic](assets/images/modeling/qlike_contribution_diagnostic.png)

## What This Means

Current proxy-stage takeaways:

- The pipeline is now beyond toy smoke tests: it has a dynamic top-50 universe,
  SEC-first event validation, an event panel, a feature matrix, model metrics,
  proxy strategy diagnostics, and figures.
- The signal-screening result is encouraging for tabular nonlinear models,
  especially LightGBM and XGBoost.
- The Mamba route is present but currently weaker than the tabular baselines.
  Because the V1 sequence drop rate is 16.3%, it remains diagnostic rather than
  a paper headline.
- No current result supports full-spread executable trading claims.

The defensible near-term claim is:

> In a no-NBBO proxy sample, state and event-history features show preliminary
> cross-sectional ranking signal for earnings event-variance mispricing beyond
> the market-implied IVAR baseline.

The paper-grade claim requires:

- historical bid/ask or NBBO-equivalent option data;
- quote-based IVAR and leg-level strategy construction;
- full bid-ask crossing as the main cost assumption;
- robustness across DTE windows, years, liquidity regimes, and BMO/AMC timing;
- clustered or bootstrap inference.

## Artifact Map

Local raw outputs:

| Purpose | Path |
| --- | --- |
| Data pipeline manifest | `artifacts/data_pipeline/data_pipeline_manifest.json` |
| Universe manifest | `artifacts/data_pipeline/universe/universe_manifest.json` |
| Dynamic calendar report | `artifacts/data_pipeline/dynamic_calendar/earnings_calendar_report.json` |
| Trade-proxy panel report | `artifacts/data_pipeline/trade_proxy_panel/trade_proxy_panel_report.json` |
| Feature matrix | `data/gold/modeling/feature_matrix.parquet` |
| Daily sequence tensor | `data/gold/modeling/sequence_tensor.npz` |
| Hybrid sequence tensor | `data/gold/modeling/hybrid_sequence_tensor.npz` |
| Proxy surface distribution audit | `artifacts/modeling/proxy_surface_distribution_audit.csv` |
| Forecast metrics | `artifacts/modeling/forecast_metrics.csv` |
| Ranking metrics | `artifacts/modeling/ranking_metrics.csv` |
| Strategy metrics | `artifacts/modeling/strategy_metrics.csv` |
| Model diagnostics | `artifacts/modeling/model_fit_diagnostics.csv` |
| Proxy report | `reports/modeling/proxy_research_report.md` |

Published docs assets:

| Purpose | Path |
| --- | --- |
| Curated results page | `docs/results_snapshot.md` |
| Published figure copies | `docs/assets/images/modeling/*.png` |

## Docs Structure

The reader-facing docs intentionally stay small:

- Home: short project overview and current status.
- Results Snapshot: current curated data/model/proxy-strategy results and
  analysis.
- Paper Plan: research protocol, target variables, model ladder, and evaluation
  design.
- Audit Prompts: implementation and manuscript review checklists.
- Future Work: paper blockers and deferred extensions.

`SPEC.md` remains the repo-root implementation contract and is not a separate
nav page.
