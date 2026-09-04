# Implementation Plan: Selection-Layer Evolution (walk-forward search)

**Branch**: `04-selection-evolution` | **Date**: 2026-08-31 | **Spec**: `specs/selection-evolution/spec.md`

**Input**: Feature specification from `specs/selection-evolution/spec.md`

## Summary

Search the selection layer (floor x ceil x chop_max grids, 80 candidates +
live baseline) on the existing 98,363-row scored-signal dataset using
walk-forward folds (contiguous 6-month folds before 2025-11-01). Select the
best candidate by robust median-fold avgR (PF>=1.8 in >=70% of folds), then
run the pre-registered hold-out verdict (ts >= 2025-11-01, same boundary as
cycle 3) with the four-bar GO rule. Output: the winning config (if any),
verdict.md, robustness tables, and a memo proposing (never auto-applying)
any live change. Everything runs on the dataset — no replay, no GPU,
~10-20 minutes.

## Technical Context

**Language/Version**: Python 3.12 (repo .venv)
**Primary Dependencies**: pandas, numpy (existing); pytest (existing)
**Storage**: reads selection_validator/data/signals_*.jsonl; writes
selection_validator/results/evolution_folds.json + specs/selection-evolution/
**Testing**: pytest — tests/test_evolution.py
**Target Platform**: Linux
**Project Type**: research harness extension (search + verdict)
**Performance Goals**: 80 candidates x 11 folds evaluated in < 20 min
**Constraints**: frozen loop/models/exit/window (FR-007); hold-out never
touched by selection (FR-003); bootstrap 10k seed 42 (FR-005)
**Scale/Scope**: 80 candidates, ~11 folds, 1 verdict

## Constitution Check

*GATE: must pass — verified against .specify/memory/constitution.md.*

| Principle | Status |
|-----------|--------|
| I. Evidence over prediction | PASS — selection-layer only, no direction model |
| II. Pre-registered experiment | PASS — grids + rule fixed in spec before code |
| III. Settled doctrine | PASS — loop/models/exit/window frozen (FR-007) |
| IV. Validation gates | PASS — walk-forward folds + hold-out bootstrap p<0.05 |
| V. Approval before build | PASS — docs to user gate first |
| VI. Fail-closed ops | PASS — memo proposes, never auto-wires |
| VII. Evidence-budget match | PASS — pure filters on existing data |
| VIII. Honest accounting | PASS — multiple-testing guard explicit; top-3 reported |

## Project Structure

```text
specs/selection-evolution/
├── spec.md, plan.md, tasks.md, verdict.md, validation-memo.md

selection_validator/
└── evolve_search.py        # NEW — evaluator, folds, selection, verdict

tests/test_evolution.py     # NEW — evaluator sanity, fold integrity, verdict
selection_validator/results/evolution_folds.json
```

## Phase Breakdown

- **Phase 0 — Evaluator**: candidate config (floor, ceil, chop_max) -> stats
  via shared evaluate() on the window-filtered dataset. Chop: cache one
  ATR14/ATR100 ratio per signal (reuse regime_halt's bars-at-ts machinery),
  thresholds derived from the cached ratio. Sanity: (0.35, 0.50, 1.0) must
  reproduce the live baseline (SC-001).
- **Phase 1 — Folds**: contiguous 6-month folds over pre-2025-11-01; assert
  no leakage (FR-003). Evaluate all candidates x all folds.
- **Phase 2 — Selection**: robust rule (FR-004) -> top-3; top-1 only decides.
  Pre-registered tiebreaks (PF, negative-fold count, grid order).
- **Phase 3 — Verdict**: top-1 vs live config on the hold-out (cycle-3
  boundary); bootstrap 10k seed 42; four-bar GO rule -> verdict.md.
- **Phase 4 — Report + memo**: fold tables JSON, validation-memo.md (ONE
  recommendation), commit, git-diff check that live files untouched.

## Reuse (no reinvention)

- selection_validator.selectors.evaluate/stats; harness.bootstrap_diff (seed 42).
- regime_halt HaltSimulator._chop_blocked / bars-at-ts caching (extract the
  ratio cache into a shared helper).
- time_window.window_mask + OOS_BOUNDARY (2025-11-01 continuity).

## Complexity Tracking

None — filters + folds + the standard verdict; no new infrastructure.
