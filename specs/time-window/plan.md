# Implementation Plan: Time-Window Gate (09:30-12:00 ET)

**Branch**: `03-time-window` | **Date**: 2026-08-31 | **Spec**: `specs/time-window/spec.md`

**Input**: Feature specification from `specs/time-window/spec.md`

## Summary

Test the user's window hypothesis (entries only US 09:30-12:00 ET) against
the all-day baseline on a fixed OOS slice (ts >= 2025-11-01), using the
existing cycle-1 dataset and the established verdict machinery. The gate is a
pure time filter — zero parameters — so the experiment is a slice + bootstrap
+ verdict, ~15 minutes of compute, no new data, no GPU.

## Technical Context

**Language/Version**: Python 3.12 (repo .venv)
**Primary Dependencies**: pandas, numpy (existing); pytest (existing)
**Storage**: reads selection_validator/data/signals_*.jsonl; writes
specs/time-window/verdict.md + validation-memo.md +
selection_validator/results/time_window_stability.json
**Testing**: pytest, tests/test_time_window.py
**Target Platform**: Linux
**Project Type**: research harness extension (measurement + verdict)
**Performance Goals**: complete in < 15 min
**Constraints**: no config/loop changes; fixed OOS boundary; seed 42;
first-hour sub-window reported but never deciding
**Scale/Scope**: 98,363 rows, 1 time filter, 1 verdict

## Constitution Check

*GATE: must pass — verified against .specify/memory/constitution.md.*

| Principle | Status |
|-----------|--------|
| I. Evidence over prediction | PASS — selection/flow change, no model |
| II. Pre-registered experiment | PASS — boundary + rule fixed in spec before code |
| III. Settled doctrine | PASS — loop frozen; window is a sanctioned flow lever |
| IV. Validation gates | PASS — OOS slice, bootstrap p<0.05 seed 42, N>=100 |
| V. Approval before build | PASS — docs shown to user first |
| VI. Fail-closed ops | PASS — memo proposes; no live edits |
| VII. Evidence-budget match | PASS — pure filter on existing data |
| VIII. Honest accounting | PASS — first-hour reported without decision power |

## Project Structure

```text
specs/time-window/
├── spec.md
├── plan.md
├── tasks.md
├── verdict.md            # written at the end
└── validation-memo.md    # ONE recommendation

selection_validator/
└── time_window.py        # NEW — window filter + comparison + verdict
tests/test_time_window.py # NEW — window boundaries + verdict rule
selection_validator/results/time_window_stability.json
```

## Phase Breakdown

- **Phase 0 — Gate**: ET minute-of-day filter [570, 720) on the dataset
  (UTC-4 for the summer-period data).
- **Phase 1 — Tests**: window boundary (11:59:59 in, 12:00:00 out), shared
  evaluator, synthetic GO / KILL (August-worse) / INCONCLUSIVE (thin).
- **Phase 2 — Verdict**: OOS slice (ts >= 2025-11-01): window vs all-day
  stats, bootstrap P(ΔavgR>0), four-condition rule -> verdict.md.
- **Phase 3 — Stability**: per-year and per-symbol tables (report only) ->
  results JSON.
- **Phase 4 — Memo + commit**: validation-memo.md (one recommendation);
  spec-kit commit; git-diff check that config/supervisor/bot untouched.

## Reuse (no reinvention)

- selection_validator.dataset (load_rows), selectors (evaluate/stats),
  harness (bootstrap_diff, seed 42) — all cycle-1 machinery.
- Time filter mirrors the live-gate semantics: entries only in window;
  management continues (out of scope here — the dataset is entries).

## Complexity Tracking

None — a filter + the standard verdict; no new infrastructure.
