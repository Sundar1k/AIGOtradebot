# Tasks: Regime-Conditioned Direction

**STATUS 2026-09-01: COMPLETE — verdict KILL. The direction question is CLOSED: post-signal 30-min momentum is REAL (65% hit, r≈0.30, P=1.0 hold-out, verified 3 levels) but does NOT improve the book (aligned -0.014 vs +0.027 avgR; Aug worse) — the funnel already captures it, costs eat the rest. No direction component; new data types are the only legitimate revisit. T001-T015 done.**

**Input**: `specs/conditioned-direction/spec.md`, `specs/conditioned-direction/plan.md`
**Prerequisites**: plan.md + spec.md (both at approval gate)

## Phase 1: Forward dataset (US1)

- [x] T001 [US1] `selection_validator/direction_audit.py`: forward K-bar
      returns (K=10, 30) for every funnel signal from data/*_3min.csv
      (FR-001, point-in-time)
- [x] T002 [US1] Spot-check: forward return == close[ts+K] - close[ts] from
      the CSV; tail signals without forward bars flagged (not silent)
- [x] T003 [P] [US1] Unit tests: forward-return correctness; no look-ahead
      in features; tail handling

## Phase 2: Conditions (US2 — fixed list, no peeking)

- [x] T004 [US2] Compute + cache per (symbol, ts), point-in-time: EMA10/30
      alignment, ADX14>=18, ATR14/100 ratio, RSI zone, hour window, r_hat
      sign, proba band (0.40-0.50 vs 0.50-0.65), 5-bar momentum
- [x] T005 [US2] Determinism test: recomputation identical; NaN conditions
      bucketed as "unknown" (reported, never merged)
- [x] T006 [P] [US2] Unit tests: each condition reproducible; direction
      prediction rules per condition (continuation vs mean-reversion) match
      the pre-registered list

## Phase 3: Selection + verdict (US3)

- [x] T007 [US3] Per condition-horizon pair x fold hit-rates (cycle-4 fold
      split); robust survival rule: hit-rate >= 0.53 in >=70% of folds,
      n>=150 per fold; primary = best median fold hit-rate
- [x] T008 [US3] Hold-out (ts >= 2025-11-01): bootstrap P(hit > 0.50) (10k,
      seed 42), n>=150
- [x] T009 [US3] Book test: conditioned book vs baseline funnel on hold-out
      (avgR AND PF) + August not worse — the hit-rate edge must prove itself
      in R (FR-004)
- [x] T010 [P] [US3] Synthetic tests: injected 55% condition -> survives;
      coin-flip -> KILL; thin -> not selected
- [x] T011 [US3] Write `specs/conditioned-direction/direction-report.md`
      (all 16 candidates transparent)

## Phase 4: Report + memo + close-out (US4)

- [x] T012 [US4] `validation-memo.md` — ONE recommendation (wire a direction
      overlay if GO / close the direction question if KILL)
- [x] T013 Commit per spec-kit rule
- [x] T014 Verify `git diff config.py supervisor.py bot.py` EMPTY (SC-004)
- [x] T015 Update tasks.md status + skill (algo-trading-bot-workflow)

## Dependencies

Phase 1 -> 2 -> 3 -> 4 sequential; no live changes at any point (FR-005).
