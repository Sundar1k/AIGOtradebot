# Tasks: Selection Validator

**STATUS 2026-08-30: COMPLETE — verdict KILL (quality-band lever archived).**
T001-T015 executed and verified (14 tests green; dataset 98,363 rows;
verdict in verdict.md). T016-T017 (feed wiring / promotion) NOT RUN — they
only apply on a GO verdict. T018 (commit) done.

**Input**: `specs/selection-validation/spec.md`, `specs/selection-validation/plan.md`
**Prerequisites**: plan.md + spec.md (both at approval gate)
**Tests**: pytest required for all non-research-harness logic; research verdicts
are scripted with deterministic seeds.

## Phase 1: Foundation (blocking)

- [x] T0 [P] [US1] Create `selection_validator/` package + `ScoredSignal`
      dataclass in `selection_validator/dataset.py` (fields per FR-001; `outcome_r`
      Optional[float])
- [x] T0 [US1] Extend `backtest.py` with `--log-signals`: append EVERY graded
      signal (TAKE and SKIP) with proba, r_hat, direction, strategy, floor,
      ceil, gate outcomes, veto_quality (where recorded) to
      `selection_validator/data/scored_signals.jsonl` — point-in-time only
- [x] T0 [P] [US1] Ledger ingestion in `dataset.py`: merge live ledger
      (evolve/lessons/veto capture) + paper ledger rows into the same schema;
      closed trades get `outcome_r`, open ones stay null
- [x] T0 [US1] Leak-check in `dataset.py` (`leak_check()`): assert no feature
      uses a bar after the signal bar; errors (not warnings) on violation;
      unit test in `tests/test_dataset.py`
- [x] T0 [P] [US1] Unit tests: schema round-trip, null-outcome handling,
      replay-row completeness — `tests/test_dataset.py`

**Checkpoint**: dataset ≥1,000 leak-checked rows → SC-001

## Phase 2: Baseline sanity (US2)

- [x] T0 [US2] `selectors.py` with shared evaluator `evaluate(signals, selector)`
      → per-signal accept/reject + WR/avgR/sumR/PF rollups
- [x] T0 [US2] `BaselineSelector` (floor 0.35, ceil 0.50, live gate stack as
      recorded in each signal row)
- [x] T0 [US2] Sanity run on pre-2026-08-01 slice: assert avgR ∈ +0.53…+0.63
      AND PF ∈ 1.96…2.26 (doctrine 0.58 / 2.11); failure = harness bug, fix
      before Phase 3 — `tests/test_selectors.py::test_baseline_reproduces_doctrine`

**Checkpoint**: baseline reproduces the known edge → SC-002

## Phase 3: Candidate (US3)

- [x] T0 [US3] `CandidateSelector(q)`: baseline funnel + veto_quality ≥ q gate
      (q=0 rows rejected for every q — per spec edge case)
- [x] T0 [US3] Measurement sweep q ∈ {5,6,7} on the historical slice → tables
      to `selection_validator/results/` (measurement only; no decision power)
- [x] T0 [P] [US3] Unit tests: q-threshold boundary (q=6 rejects quality 5,
      accepts 6) — `tests/test_selectors.py`

## Phase 4: Verdict (US4)

- [x] T0 [US4] `harness.py`: one-sided bootstrap ΔavgR (10k draws, seed 42,
      reuse `edge_monitor.bootstrap_p_lt`); PF comparison; slice splitter with
      fixed OOS boundary `entry ts ≥ 2026-08-01`
- [x] T0 [US4] Unit tests (synthetic): injected edge → GO; noise → KILL;
      N<30 → INCONCLUSIVE — `tests/test_verdict.py`
- [x] T0 [US4] `verdict.py`: writes `specs/selection-validation/verdict.md`
      with per-slice N, WR, avgR, PF, ΔavgR, p-value + GO/KILL/INCONCLUSIVE —
      numbers only, no prose-only verdicts
- [x] T0 [US4] Full run: replay → ingest → baseline sanity → candidate sweep
      → OOS verdict; record result in verdict.md

**Checkpoint**: SC-003 — numbers-backed decision at N≥150 OOS (or INCONCLUSIVE)

## Phase 5: Forward feed (US5)

- [ ] T0 [US5] `AUTOTRADE_SELECTOR=feed` in `paper_live.py`: log
      `selector: ACCEPT/REJECT (q=…)` per signal; entries byte-identical
      (control-run comparison in the task's verification)
- [ ] T0 [US5] Forward OOS accumulation + GO-to-gate promotion proposal
      (never auto-enabled) — verdict.md appendix

## Phase 6: Polish

- [x] T0 [P] Commit per task (spec-kit rule); docs updated; archive/verdict
      recorded per constitution Principle II

## Dependencies & Execution Order

- Phase 1 → 2 → 3 → 4 strictly sequential (each checkpoint gates the next).
- Phase 5 parallel-safe with 4 (feed wiring touches paper_live.py only) but
  promoted only after verdict.
- Phase 6 after all desired phases.
- [P] tasks are file-disjoint; run in parallel if desired.
