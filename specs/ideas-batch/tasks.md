# Tasks: Ideas Batch

**STATUS 2026-09-01: COMPLETE. Gap/jump/volume filters KILLED (no signal). First-hour narrowing INCONCLUSIVE (n=89<150, P=0.554) but August-resilient (+0.51 vs -1.22) — next re-test when hold-out reaches 150. Full window stays. T001-T009 done.**

**Input**: `specs/ideas-batch/spec.md`, `specs/ideas-batch/plan.md`

- [x] T001 [US1] Feature caches per signal (point-in-time, deterministic):
      gap direction, jump flag (1.5x ATR20, last 2 bars), volume ratio,
      first-hour flag
- [x] T002 [US1] Spot-check features against raw CSV values
- [x] T003 [P] [US1] Unit tests: feature determinism, no look-ahead
- [x] T004 [US2] Candidate books: gap-aligned, gap-opposed, jump-excluded,
      volume-confirmed, first-hour-only — each ON TOP of the current funnel
- [x] T005 [US2] Train-fold tables (report only) + hold-out decision per
      variant with the common four-part bar (P>0.95, PF, n>=150, Aug)
- [x] T006 [P] [US2] Unit tests: bar logic synthetic (GO/KILL/INCONCLUSIVE)
- [x] T007 [US3] Write results/ideas_batch.json + specs/ideas-batch/report.md
- [x] T008 [US3] validation-memo.md — ONE recommendation per winning idea
      (or "none add anything")
- [x] T009 Commit; verify git diff config/supervisor/bot EMPTY; tick tasks;
      update skill

## Dependencies

Sequential 1 -> 2 -> 3. No live changes (FR-005).
