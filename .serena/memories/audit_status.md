# Audit Status

Last synced: 2026-06-12.

## Current Verification

- Actual repo root is
  `/home/tycheng/projects/earnings-event-vol/earnings-event-vol`.
- Previous `just check` passed after the completion-gap audit, quote-aware
  diagnostics, targeted quote REST worker/cache integrity wiring, 502-event
  bounded quote consolidation, sequence-all refresh, report regeneration, and
  docs sync.
- Cleanup verification reran ruff, targeted model/CLI/report tests, MkDocs
  strict build, and `git diff --check`; full `just check` was not rerun after
  this cleanup and remains the handoff gate.
- Full handoff gate result: 153 tests passed, total coverage 95.03%, ruff
  format/check passed, mypy passed, MkDocs strict build passed,
  `_check-doc-figures` passed, CLI `status` passed, and `source-probe all`
  passed.
- Remaining warnings are sklearn `n_alphas` deprecation warnings from test
  dependencies; they are non-blocking.

## Current Artifact Audit

- Active data manifest is green: `artifacts/data_pipeline/data_pipeline_manifest.json`
  has `ok=true`.
- Active research manifest is green:
  `artifacts/modeling/research_manifest.json` has `ok=true`,
  `stage=models`, `sequence_suite=all`, `bootstrap_iter=0`,
  `reuse_tuning_params=true`, and no sequence seed/backend fields after
  cleanup. It predates the canonical `tuned_phase1_day_c2c_rank_log_rvar`
  log-target profile and dual-output ensemble, and should be refreshed before
  citing current-spec model results.
- Current report manifest is stale pending report refresh:
  `artifacts/modeling/research_report_manifest.json` has `ok=true`,
  `stage=report`, and `sequence_suite=all`.
- Completion-gap audit is populated:
  `artifacts/modeling/completion_gap_audit.json` has `ok=false`,
  `paper_grade_ready=false`, 12 rows, and status counts `complete=8`,
  `diagnostic_only=2`, `incomplete=2`.
- Current docs figures were regenerated and synced under
  `docs/assets/images/modeling/`.

## Current Docs Audit

- `docs/paper_plan.md` is the paper-style manuscript plan.
- `docs/results_snapshot.md` is the paper-style Results and Discussion ledger.
- README, paper plan, results snapshot, and current Serena memories now share
  the same current-data口径: 816 events, 807 C2C RVAR rows, 705 IVAR rows, and
  10,921,438 matched quote rows in the bounded quote diagnostic slice, with
  7,183 bounded quote-IV leg rows and 471 finite bounded surface-IVAR mid rows.
- The PR #1 Chinese contributor docs are no longer present in `docs/` or
  MkDocs nav; their useful ideas were absorbed into paper-facing docs and code.

## Current Legacy Audit

- `fe_v1_legacy` remains intentionally as a feature-schema ablation, not as
  active default evidence.
- Retired legacy sequence ids remain only in retirement manifests/tests and are
  not active model ids or runtime dependencies.
- Historical ablation artifacts remain under ignored `artifacts/modeling_ablations/`.
  Treat them as historical until rerun on the current panel.
- The current result set is not final paper-grade evidence because target-window
  historical coverage and full bid/ask/NBBO-equivalent execution are still
  pending. Bounded matched quote rows, bounded quote-IV surface diagnostics,
  quote-confidence stratified results, robustness summaries, and the
  lightweight sequence diagnostic suite are populated from the latest artifact
  refresh. Mask-only/time-shuffle rows need a model/report rerun after the
  runtime cleanup before they represent current code.
