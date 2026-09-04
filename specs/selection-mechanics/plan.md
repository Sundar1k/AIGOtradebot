# Implementation Plan: Selection-Mechanics Audit

**Branch**: `05-selection-mechanics` | **Date**: 2026-09-01 | **Spec**: `specs/selection-mechanics/spec.md`

**Input**: Feature specification from `specs/selection-mechanics/spec.md`

## Summary

Dataset-only audit of the live selection mechanics: (A) a dual-fire census
(how often strategies co-fire, direction agreement, and how the max-proba
pick performed on those events) and (B) an active-lane audit (per symbol x
lane, full + hold-out stats vs symbol/book averages) with a pre-registered
removal rule, memo-gated. No replay, no GPU, no live changes. If the census
shows the winner-pick knob matters, or a lane is flagged, the memo proposes
the follow-up (dual-book replay / removal) — decided by the user.

## Technical Context

**Language/Version**: Python 3.12 (repo .venv)
**Primary Dependencies**: pandas, numpy (existing); pytest (existing)
**Storage**: reads selection_validator/data/signals_*.jsonl; writes
specs/selection-mechanics/{audit-report,validation-memo}.md
**Testing**: pytest — tests/test_mechanics.py
**Target Platform**: Linux
**Project Type**: research audit (dataset analysis)
**Performance Goals**: complete in < 15 min
**Constraints**: FR-002 hold-out boundary; FR-004 removal rule; FR-005 no
live changes; FR-006 no winner-pick change this cycle
**Scale/Scope**: 98,363 rows, 7 active lanes, 1 census, 1 memo

## Constitution Check

*GATE: must pass — verified against .specify/memory/constitution.md.*

| Principle | Status |
|-----------|--------|
| I. Evidence over prediction | PASS — selection-mechanics measurement only |
| II. Pre-registered experiment | PASS — removal rule + boundaries fixed in spec |
| III. Settled doctrine | PASS — loop frozen; audit only; override config stands |
| IV. Validation gates | PASS — hold-out bootstrap p<0.05, n>=30 |
| V. Approval before build | PASS — docs to user gate first |
| VI. Fail-closed ops | PASS — memo proposes; no live edits |
| VII. Evidence-budget match | PASS — dataset-only; expensive sims deferred/gated |
| VIII. Honest accounting | PASS — dataset limits explicit (no YM-orb outcomes) |

## Project Structure

```text
specs/selection-mechanics/
├── spec.md, plan.md, tasks.md, audit-report.md, validation-memo.md

selection_validator/
└── mechanics_audit.py     # NEW — census + lane audit + removal rule

tests/test_mechanics.py    # NEW — census determinism, removal rule, boundaries
```

## Phase Breakdown

- **Phase 0 — Census (US1)**: dual-fire events (same symbol+ts, both
  in-band at live floor/ceil 0.40-0.65, window-filtered): count, fraction of
  taken trades, direction-agreement rate, chosen-pick WR/avgR vs single-fire.
- **Phase 1 — Lane audit (US2)**: per symbol x lane full + hold-out
  n/WR/avgR/PF vs symbol and book averages; flag hold-out-negative lanes.
- **Phase 2 — Removal rule (US3)**: flag iff hold-out avgR < 0 AND n>=30 AND
  bootstrap P(avgR < symbol avgR) > 0.95 (10k, seed 42). Memo-gated.
- **Phase 3 — Report + memo (US4)**: audit-report.md + validation-memo.md
  (ONE recommendation), commit, git-diff check, tasks ticked.

## Reuse (no reinvention)

- selection_validator.dataset (load_rows), selectors (stats), harness
  (bootstrap_diff seed 42), time_window (window_mask + boundary).
- The audit mirrors the LIVE mechanism (floor/ceil 0.40/0.65) per FR-003.

## Complexity Tracking

None — two table analyses + the standard bootstrap rule.
