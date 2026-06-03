# Research Status 2026-05-12

Latest local state for `earnings-event-vol` after the canonical FE V2 tuned run
and same-code FE V1 versus FE V2 ablation.

## Verification State

- Canonical active command:
  `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v2_sec_xbrl"`.
- FE V1 ablation command:
  `just research args="--stage all --sequence-suite all --allow-high-sequence-risk --bootstrap-iter 1000 --tuning-profile tuned_phase1 --feature-schema-version fe_v1_legacy"`.
- Active artifacts were restored to FE V2 after the FE V1 ablation.
- Ablation snapshots are saved under
  `artifacts/modeling_ablations/fe_v2_sec_xbrl/` and
  `artifacts/modeling_ablations/fe_v1_legacy/`.
- Active `research_manifest.json` reports `ok=true`, `sequence_suite=all`,
  `bootstrap_iter=1000`, `tuning_profile=tuned_phase1`, and
  `feature_schema_version=fe_v2_sec_xbrl`.
- Metrics and tuning artifacts now carry both `feature_schema_version` and
  `tuning_profile`.

## Current Data State

- Current proxy run window: 2022-12-01 through 2025-12-31.
- Dynamic calendar rows: 1,054.
- BMO/AMC main-sample events: 810.
- Events with C2C `rvar_event` alias: 801.
- Events with trade-proxy IVAR: 693.
- Proxy contracts: 12,038.
- Contracts with usable pre-cutoff proxy price: 10,165.
- Contracts with no trade in cutoff window: 1,873.
- Contracts with local IV proxy: 10,138.
- Main DTE 5-14 contracts: 5,098.
- Robustness DTE 3-21 contracts: 12,038.
- Proxy straddle diagnostics rows: 779.
- Panel grade remains `no_nbbo_trade_proxy`; `paper_grade=false`.

## Current Research Package

- Feature rows: 810.
- Prediction rows: 2,430.
- Trained model-target fits: 33.
- FE V2 event-level model features: 243.
- FE V2 tree model features: 397.
- Default schema: `fe_v2_sec_xbrl`.
- Ablation schema: `fe_v1_legacy`.
- Canonical protocol: `tuned_phase1`, train/validation-only selection,
  train+validation refit, locked test evaluated once.

## Research Question and Target System

The paper-facing question is whether models improve trading decisions around
option-implied earnings event variance mispricing. This is not generic IV
forecasting.

- `jump_c2o`: primary scientific target, close-to-open earnings jump variance.
- `day_c2c`: literature-compatible target and the only V1 proxy-PnL headline.
- `reaction_o2c`: post-open digestion diagnostic.
- Market baseline: `IVAR_event`.
- C2C ex post mispricing: `RVAR_event_day_c2c - IVAR_event`.
- Trade decisions use premium-space expected edge, not raw variance edge.

## Sequence Status

- Active hybrid sequence tensor: `31 x 21`, with 19 prior daily steps plus
  12 entry-day five-minute trade-aggregate proxy bins.
- Daily sequence eligible events: 678 out of 810.
- Default sequence drop rate: 16.3%; `high_sequence_selection_risk=true`.
- Hybrid sequence is not sparse: 682 events have at least eight valid intraday
  bins, median hybrid mask density is 0.7419, and
  `hybrid_sequence_too_sparse=false`.
- Legacy fake Mamba ids are retired because they were in-repo gated-RNN
  variants, not official `mamba-ssm`.
- The full sequence suite did not pass the diagnostic-grade sequence gate.
  Do not sell Mamba unless the sample/data route changes materially.

## Current Results

Active FE V2 default:

- `jump_c2o`: best AUC is Goyal-Saretto spread, about 0.602; best OOS R2
  versus IVAR is LightGBM, about 0.203.
- `day_c2c`: best headline proxy PnL is ridge-flat sequence aggregates, about
  19,918 USD, but this is diagnostic because sequence risk remains high.
- Tuned LightGBM/XGBoost FE V2 rows do not provide the current sell.

Same-code FE V1 ablation:

- `jump_c2o`: LightGBM best AUC about 0.677; XGBoost best OOS R2 versus IVAR
  about 0.375.
- `day_c2c`: LightGBM best headline proxy PnL about 53,664 USD; XGBoost best
  OOS R2 versus IVAR about 0.574.
- `reaction_o2c`: ridge-flat sequence best AUC about 0.799; XGBoost best OOS
  R2 about 0.949. O2C remains diagnostic.

## Conservative Results and Limitations

The strongest proxy-stage evidence is not generic RMSE improvement and not FE
V2 richness. It is that a parsimonious FE V1 tabular feature set has preliminary
cross-sectional ranking signal and maps into positive `day_c2c` premium-space
proxy economics under a no-NBBO proxy cost model. FE V2 is currently a negative
diagnostic result.

Do not claim paper-grade executable performance, Mamba superiority,
full-spread tradability, NBBO evidence, FE V2 improvement, or that lower RMSE
alone proves economic value. Paper-grade claims require historical quote/NBBO
or equivalent data, quote-based IVAR, leg-level execution with realistic
bid/ask crossing, DTE and liquidity robustness, and clustered or bootstrap
inference.
