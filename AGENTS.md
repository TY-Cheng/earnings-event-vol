# Agent Instructions

This repository is `earnings-event-vol`, a research pipeline for earnings event
variance forecasting and risk-defined option backtests.

## Project Contract

- Treat `SPEC.md` as the implementation and research-protocol contract.
- Do not reframe the project as generic implied-volatility forecasting.
- The paper-facing question is whether models improve trading decisions around
  option-implied earnings event variance mispricing.
- Models forecast `RVAR_event`; ex post mispricing is `RVAR_event - IVAR_event`;
  trade entry uses premium-space expected edge, not raw variance edge.

## Workflow

- Use `just` as the command surface.
- Run `just check` before handing off code changes.
- `just check` formats with `ruff format`, fixes lint with `ruff check --fix`,
  then runs mypy, pytest, MkDocs strict build, status, and source probes.
- Keep the public command surface small. Prefer parameterized `just data ...`
  variants over adding new top-level recipes.

## Data And Secrets

- Keep `.env` machine-local and ignored.
- Do not print, commit, or inline Massive keys.
- Massive credentials must be file paths such as `MASSIVE_API_KEY_FILE` and
  `MASSIVE_FLAT_FILE_KEY_FILE`.
- Generated data and artifacts stay under ignored `data/`, `artifacts/`,
  `reports/`, or `site/` paths unless explicitly curated.
- Label second-aggregate or trade-price proxy outputs as `no_nbbo_trade_proxy`;
  do not describe them as bid/ask or NBBO-executable backtests.

## Documentation

- The docs site has five reader-facing entries: Home, Results Snapshot, Paper
  Plan, Audit Prompts, and Future Work.
- Keep `SPEC.md` as a repo-root protocol file, not as a separate docs-nav page.
- Keep paper claims conservative: lower RMSE is not enough unless the signal
  improves tradable ranking and net performance after realistic costs.
