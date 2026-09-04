# Implementation Plan: Regime-Halt Validation

**Branch**: `02-regime-halt` | **Date**: 2026-08-31 | **Spec**: `specs/regime-halt/spec.md`

**Input**: Feature specification from `specs/regime-halt/spec.md`

## Summary

Replay the two LIVE halt rules (edge-monitor realized-edge halt + chop gate)
point-in-time over the existing 5.3y scored-signal dataset (98,363 rows —
already built by cycle 1, no new replay, no GPU). Book-level chronological
simulation: maintain the trailing closed-trade window, apply the edge rule
and chop gate at each signal, produce a candidate trade subset, compare vs
the baseline funnel on the full period AND the August 2026 slice, and emit
the pre-registered verdict plus a keep/adjust/remove memo for the live
config. Parameters are the LIVE values — zero tuning (FR-004).

## Technical Context

**Language/Version**: Python 3.12 (repo .venv)
**Primary Dependencies**: pandas, numpy (existing); pytest 9.1.1 (existing)
**Storage**: reads `selection_validator/data/signals_*.jsonl`; writes
`selection_validator/results/` + `specs/regime-halt/verdict.md` + memo
**Testing**: pytest, tests/ (reuse conftest style)
**Target Platform**: Linux (this box)
**Project Type**: research harness (extension of selection_validator)
**Performance Goals**: full-dataset simulation + bootstrap in < 15 min
**Constraints**: point-in-time only; fixed seed 42; live parameters only;
no changes to config.py / supervisor.py / bot.py; no GPU
**Scale/Scope**: 98,363 signals, 1 state machine, 2 rules, 1 verdict

## Constitution Check

*GATE: must pass — verified against .specify/memory/constitution.md.*

| Principle | Status |
|-----------|--------|
| I. Evidence over prediction | PASS — selection-only, no direction model |
| II. Pre-registered experiment | PASS — success bar + kill rule in spec before code |
| III. Settled doctrine | PASS — loop/config frozen (FR-006) |
| IV. Validation gates | PASS — point-in-time, bootstrap p<0.05 seed 42, live params only |
| V. Approval before build | PASS — this plan + spec + tasks go to the user gate first |
| VI. Fail-closed ops | PASS — memo proposes, never auto-edits live config |
| VII. Evidence-budget match | PASS — reuses cycle-1 dataset + harness; one new module |
| VIII. Honest accounting | PASS — regime-HMM limitation documented, no proxy |

## Project Structure

```text
specs/regime-halt/
├── spec.md              # Approved gate
├── plan.md              # This file
├── tasks.md             # Next gate
├── verdict.md           # Numbers-backed decision (written at the end)
└── validation-memo.md   # ONE recommendation for the live config

selection_validator/
├── regime_halt.py       # NEW — point-in-time halt simulator + comparison
└── (reuse dataset.py, selectors.py, harness.py, verdict machinery)

tests/
└── test_regime_halt.py  # NEW — state-machine transitions + GO/KILL/INC

selection_validator/results/
└── regime_halt_sensitivity.json   # US3 measurement tables (no decision power)
```

## Phase Breakdown

- **Phase 0 — Simulator**: `regime_halt.py` implements the edge-monitor state
  machine (normal/watch/halt/24h-cooldown-resume) + chop gate, book-level,
  chronological over the dataset. Cold start = take everything (<15 closed).
- **Phase 1 — Tests**: unit tests for streak→halt, cooldown, resume, chop
  block, cold start; synthetic GO/KILL/INCONCLUSIVE for the verdict rule.
- **Phase 2 — Comparison**: candidate vs baseline stats (shared evaluate
  path) on full replay + August slice; halt-fraction; max drawdown of sumR.
- **Phase 3 — Sensitivity (train slice only)**: COOLDOWN 12/24/48h and
  HALT_P 0.85/0.90/0.95 variants → JSON tables, recorded, never deciding.
- **Phase 4 — Verdict + memo**: pre-registered GO/KILL/INCONCLUSIVE;
  `verdict.md` with all four GO conditions shown; `validation-memo.md` with
  one keep/adjust/remove recommendation.
- **Phase 5 — Commit**: spec-kit rule (commit per task); git diff check that
  config.py/supervisor.py/bot.py are untouched.

## Reuse (no reinvention)

- `selection_validator.dataset` — load_rows, leak_check, merge helpers.
- `selection_validator.selectors` — shared evaluate() (FR-002).
- `selection_validator.harness` — bootstrap_diff (10k, seed 42), split_slices.
- `edge_monitor.py` — copy the state-machine SEMANTICS into the simulator
  (read-only reference; the live module itself is not modified).
- `chop_gate.py` — call `atr_ratio` / `should_block` directly (deterministic,
  importable, point-in-time by construction).

## Complexity Tracking

None — no constitution violations. One new module + tests; everything else is
reuse of cycle-1 machinery.
