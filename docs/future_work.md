# Future Work

Keep the first paper narrow. Extensions should not enter the headline until the
event-variance target, timestamp alignment, baselines, and two risk-defined
strategies are reproducible.

## Phase 2 Extensions

Calendar straddles:

- Add only after long straddle and short iron fly backtests are stable.
- Treat as a relative-value strategy, not a pure event variance bet.
- Vega-normalize at entry and report residual Greeks.

Intraday execution:

- Use OPRA or another intraday quote source only after the EOD design is audited.
- Study 15:45 or 15:59 entry, open-auction exit, and first 5/30/60 minute IV
  crush dynamics.

Richer event calendars:

- Add multiple vendor calendars only if timestamp disagreements are auditable.
- Preserve BMO/AMC as the first paper sample; DMH remains excluded unless the
  execution design is intraday.

Model extensions:

- Add conformal prediction or distributional calibration after quantile models
  are stable.
- Compare Mamba to a compact temporal Transformer only after LightGBM and
  FT-Transformer are fully tuned.

Cross-sectional portfolio construction:

- Add volatility-budgeted allocation.
- Cap ticker, sector, and earnings-date concentration.
- Report capital-at-risk and premium-at-risk separately.

## Do Not Add Yet

- Naked short straddles.
- Unbounded short gamma strategies.
- Hand-repaired earnings timestamps.
- Vendor proprietary alpha features that cannot be separated from model leakage.
- A manuscript claim based only on IV RMSE.
