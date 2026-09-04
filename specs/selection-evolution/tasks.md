# Tasks: Selection-Layer Evolution (walk-forward search)

**STATUS 2026-08-31: COMPLETE — verdict INCONCLUSIVE (140 hold-out trades < 150 pre-registered bar; all quality bars passed: P=0.9954, PF 1.82 vs 0.50, Aug better). Winner (0.40/0.65/2.0) positive in 10/10 folds. Live config stays; re-run when hold-out grows. T001-T017 done.**

**Input**: `specs/selection-evolution/spec.md`, `specs/selection-evolution/plan.md`
**Prerequisites**: plan.md + spec.md (both at approval gate)

## Phase 1: Evaluator (US1)

- [x] T001 [US1] `selection_validator/evolve_search.py`: CandidateConfig
      (floor, ceil, chop_max) -> stats via shared evaluate() on the
      window-filtered dataset (FR-001)
- [x] T002 [US1] Chop ratio cache: one ATR14/ATR100 per signal (trailing 300
      bars <= ts, point-in-time); all thresholds derived from the cache
      (FR-002 — extract the shared helper from regime_halt)
- [x] T003 [US1] Sanity test: (0.35, 0.50, 1.0) reproduces the live baseline
      (+0.43R / PF 1.80 full-history, within tolerance) — SC-001
- [x] T004 [P] [US1] Unit tests: floor-0.40 filter matches a manual filter;
      chop thresholds 0.8/1.0/1.2 derived from one cached ratio

## Phase 2: Folds + selection (US2)

- [x] T005 [US2] Contiguous 6-month folds over pre-2025-11-01; assert no
      overlap/leakage (FR-003)
- [x] T006 [US2] Evaluate 80 candidates x all folds -> per-candidate fold table
- [x] T007 [US2] Robust selection (FR-004): best median fold avgR among
      PF>=1.8 in >=70% of folds; tiebreaks (PF, negative-fold count, grid
      order); top-3 out, top-1 decides
- [x] T008 [P] [US2] Unit tests: fold integrity; lucky-in-one-fold candidate
      does NOT win; empty-fold handling

## Phase 3: Hold-out verdict (US3)

- [x] T009 [US3] Top-1 vs live config on hold-out (ts >= 2025-11-01);
      bootstrap P(ΔavgR>0) (10k, seed 42)
- [x] T010 [US3] Four-bar rule: (a) P>0.95 (b) PF_cand>=PF_live (c) Aug
      avgR_cand>avgR_live (d) n_cand>=150 -> GO / KILL / INCONCLUSIVE
- [x] T011 [P] [US3] Synthetic tests: real hold-out edge -> GO; train-lucky
      -> not GO; thin -> INCONCLUSIVE
- [x] T012 [US3] Write `specs/selection-evolution/verdict.md` (four bars shown)

## Phase 4: Report + memo + close-out (US4)

- [x] T013 [US4] Per-fold tables for top-3 -> results/evolution_folds.json
      (report only; candidates 2-3 never decide)
- [x] T014 [US4] `validation-memo.md` — ONE recommendation (wire new config /
      keep live config)
- [x] T015 Commit per spec-kit rule
- [x] T016 Verify `git diff config.py supervisor.py bot.py` EMPTY (SC-004)
- [x] T017 Update tasks.md status + skill (algo-trading-bot-workflow)

## Dependencies

Phase 1 -> 2 -> 3 -> 4 sequential; tests within each phase. No live changes
at any point (FR-006).
