# Implementation Plan: Regime-Conditioned Direction

**Branch**: `06-conditioned-direction` | **Date**: 2026-09-01 | **Spec**: `specs/conditioned-direction/spec.md`

**Input**: Feature specification from `specs/conditioned-direction/spec.md`

## Summary

Build the forward-K-bar direction dataset for every funnel signal, compute 8
pre-registered point-in-time conditions x 2 horizons (16 candidates), measure
conditional hit-rates across the cycle-4 walk-forward folds, robustly select
survivors (>=0.53 in >=70% of folds, n>=150), then run the four-bar hold-out
verdict — including the book test (does trading the condition improve the
funnel?). Outcome: the direction question is answered with numbers — a
survivor becomes a candidate direction overlay; no survivor closes the
question permanently. Dataset + bar computation only — no replay, no GPU,
~30-60 minutes.

## Technical Context

**Language/Version**: Python 3.12 (repo .venv)
**Primary Dependencies**: pandas, numpy (existing); pytest (existing)
**Storage**: reads selection_validator/data/signals_*.jsonl + data/*_3min.csv;
writes specs/conditioned-direction/ + selection_validator/results/
**Testing**: pytest — tests/test_direction.py
**Target Platform**: Linux
**Project Type**: research audit (direction measurement)
**Performance Goals**: features + forward returns for ~11k signals in < 20 min
**Constraints**: FR-001 point-in-time; FR-002 fixed list/horizons; FR-003
folds/hold-out; FR-005 no live changes
**Scale/Scope**: 11,364 signals, 8 conditions x 2 horizons, 1 verdict

## Constitution Check

*GATE: must pass — verified against .specify/memory/constitution.md.*

| Principle | Status |
|-----------|--------|
| I. Evidence over prediction | PASS — measures the prediction question, decides on the book |
| II. Pre-registered experiment | PASS — condition list + bar fixed at spec time |
| III. Settled doctrine | PASS — loop frozen; a survivor would be a NEW validated overlay |
| IV. Validation gates | PASS — folds + hold-out bootstrap p<0.05, n>=150 |
| V. Approval before build | PASS — docs to user gate first |
| VI. Fail-closed ops | PASS — memo proposes, never auto-wires |
| VII. Evidence-budget match | PASS — dataset + bars only, no replay/GPU |
| VIII. Honest accounting | PASS — all 16 candidates reported; book test gates |

## Project Structure

```text
specs/conditioned-direction/
├── spec.md, plan.md, tasks.md, direction-report.md, validation-memo.md

selection_validator/
└── direction_audit.py      # NEW — forward returns, conditions, hit-rates,
                            #        selection, verdict

tests/test_direction.py     # NEW — forward-return correctness, condition
                            #        determinism, selection/verdict rules
```

## Phase Breakdown

- **Phase 0 — Forward dataset (US1)**: forward K-bar returns (K=10, 30) for
  every funnel signal from the bar CSVs; spot-check against the CSVs.
- **Phase 1 — Conditions (US2)**: the 8 pre-registered conditions, computed
  point-in-time and cached per (symbol, ts): EMA alignment, ADX, vol ratio,
  RSI zone, hour window, r_hat sign, proba band, momentum.
- **Phase 2 — Selection (US3)**: per condition-horizon pair, per-fold
  hit-rates (cycle-4 fold split); robust survival rule (>=0.53 in >=70% of
  folds, n>=150/fold); primary = best median fold hit-rate.
- **Phase 3 — Verdict (US3)**: hold-out bootstrap P(hit>0.50) (10k, seed 42)
  + n>=150 + conditioned-book vs baseline funnel (avgR AND PF) + August not
  worse -> GO / KILL / INCONCLUSIVE.
- **Phase 4 — Report + memo (US4)**: all 16 candidates' tables,
  direction-report.md, validation-memo.md (ONE recommendation), commit,
  git-diff check, tasks ticked.

## Reuse (no reinvention)

- selection_validator.dataset (load_rows), harness (bootstrap_diff seed 42),
  time_window (window_mask), evolve_search.make_folds (cycle-4 folds),
  mechanics_audit bar-at-ts caching pattern.

## Complexity Tracking

None — forward returns + feature cache + the standard fold/bootstrap verdict.
