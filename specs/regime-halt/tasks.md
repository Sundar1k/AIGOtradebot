# Tasks: Regime-Halt Validation

**STATUS 2026-08-31: COMPLETE — verdict KILL for the current config (over-halts
64% vs the 50% pre-registered bar), but the rules demonstrably catch regimes
(Aug -587R -> -20R) and improve per-trade quality (PF 1.80 -> 1.99). Memo
recommends ADJUST via a new pre-registered chop-threshold experiment.
T001-T014 executed, tests green (160 suite-wide); T015-T017 done.

**Input**: `specs/regime-halt/spec.md`, `specs/regime-halt/plan.md`
**Prerequisites**: plan.md + spec.md (both at approval gate)
**Tests**: pytest required (state machine + verdict rule).

## Phase 1: Simulator (US1)

- [x] T001 [US1] `selection_validator/regime_halt.py`: book-level chronological
      simulator over the dataset — signal order by ts across symbols
- [x] T002 [US1] Edge-monitor state machine, EXACT live semantics:
      trailing window N=15 closed trades; halt iff P(meanR<0)>0.90 (bootstrap
      10k, seed 42) OR (WR<0.30 AND meanR<0); 24h cooldown then resume-to-test
      with fresh window; re-halt if still losing (mirror edge_monitor.py)
- [x] T003 [US1] Chop gate: `chop_gate.should_block(bars_at(ts))` per signal
      (trailing 300 bars <= ts, ATR14/ATR100 >= 1.0, MIN_BARS=120, fail-open)
- [x] T004 [US1] Cold start (<15 closed trades): take everything
- [x] T005 [P] [US1] Unit tests: streak->HALT, cooldown blocks at T+2h and
      evaluates at T+26h, chop block, cold start — `tests/test_regime_halt.py`

**Checkpoint**: simulator reproduces the live state machine → SC-001

## Phase 2: Comparison (US2)

- [x] T006 [US2] Candidate subset = funnel trades surviving both rules;
      stats via shared `evaluate()` — full replay + August 2026 slice
- [x] T007 [US2] Metrics: n, WR, avgR, sumR, PF, halt-fraction, max sumR
      drawdown (candidate vs baseline)
- [x] T008 [P] [US2] Unit tests: identical evaluator for both sides (FR-002)

**Checkpoint**: both periods computed → comparison ready

## Phase 3: Sensitivity (US3 — measurement only, train slice)

- [x] T009 [US3] COOLDOWN {12,24,48}h and HALT_P {0.85,0.90,0.95} variants on
      pre-2026-08-01 slice -> `selection_validator/results/regime_halt_sensitivity.json`
- [x] T010 [US3] Assert variants never enter the verdict rule (constitution II)

## Phase 4: Verdict + memo (US4, US5)

- [x] T011 [US4] Bootstrap P(ΔavgR>0) (10k, seed 42) + full pre-registered rule:
      GO iff P>0.95 AND PF_cand>=PF_base AND Aug avgR_cand>avgR_base AND
      n_cand>=0.5*n_base; else KILL/INCONCLUSIVE per spec
- [x] T012 [US4] Unit tests: synthetic wiped-out window -> GO; helps-overall-
      but-not-August -> KILL (condition c); halts-everything -> KILL (d)
- [x] T013 [US4] Write `specs/regime-halt/verdict.md` (all four bars shown)
- [x] T014 [US5] Write `specs/regime-halt/validation-memo.md` — ONE
      keep/adjust/remove recommendation for the live config (proposal only)

**Checkpoint**: SC-002 + SC-003

## Phase 5: Close-out

- [x] T015 Commit per task (spec-kit rule)
- [x] T016 Verify `git diff config.py supervisor.py bot.py` is EMPTY (SC-004)
- [x] T017 Update `specs/regime-halt/tasks.md` status header like cycle 1

## Dependencies & Execution Order

- Phase 1 -> 2 -> 4 strictly sequential (checkpoints gate).
- Phase 3 parallel-safe with 2 (train slice only).
- Phase 5 last. No live config changes at any point.
