# Tasks: Time-Window Gate (09:30-12:00 ET)

**STATUS 2026-08-31: COMPLETE — verdict GO (first GO in three cycles).
OOS (Nov 2025-Aug 2026): window +0.455R/PF 1.93 vs all-day +0.027R/PF 1.03,
P(ΔavgR>0)=1.0 (DST-corrected — user caught the fixed-offset bug), all four
pre-registered bars passed. WIRED INTO LIVE 2026-08-31 (user-approved):
autotrade.service + paper-trade.service run AUTOTRADE_ENTRY_WINDOW=1;
entries only 09:30-12:00 ET (America/New_York), management 24h. T001-T014 done.

**Input**: `specs/time-window/spec.md`, `specs/time-window/plan.md`
**Prerequisites**: plan.md + spec.md (both at approval gate)

## Phase 1: Gate + tests (US1)

- [x] T001 [US1] `selection_validator/time_window.py`: ET minute-of-day per
      signal (UTC-4); window mask [570, 720) — half-open, FR-001
- [x] T002 [US1] Candidate = window mask on funnel trades; baseline = all
      funnel trades; shared evaluate() path (FR-002)
- [x] T003 [P] [US1] Tests: boundary (11:59:59 in / 12:00:00 out), identical
      evaluator — `tests/test_time_window.py`

## Phase 2: Verdict (US2)

- [x] T004 [US2] OOS slice: ts >= 2025-11-01 (FR-004); window vs all-day
      stats on OOS; bootstrap P(ΔavgR>0) (10k, seed 42)
- [x] T005 [US2] Four-condition rule: (a) P>0.95 (b) PF_w>=PF_a (c) Aug
      avgR_w > avgR_a (d) n_w>=100 -> GO / KILL / INCONCLUSIVE
- [x] T006 [P] [US2] Synthetic tests: GO, KILL (August worse), INCONCLUSIVE
      (thin) — `tests/test_time_window.py`
- [x] T007 [US2] Write `specs/time-window/verdict.md` (all four bars shown)

## Phase 3: Stability report (US3 — no decision power)

- [x] T008 [US3] Per-year (2021-2026) and per-symbol window-vs-all-day tables
      -> `selection_validator/results/time_window_stability.json`
- [x] T009 [US3] First-hour (09:30-10:30) sub-window stats appended as
      secondary observation — explicitly non-deciding

## Phase 4: Memo + close-out (US4)

- [x] T010 [US4] `specs/time-window/validation-memo.md` — ONE recommendation
      (wire gate into live supervisor entries / keep as-is / drop)
- [x] T011 Commit per spec-kit rule
- [x] T012 Verify `git diff config.py supervisor.py bot.py` EMPTY (SC-004)
- [x] T013 Update tasks.md status header + skill (algo-trading-bot-workflow)

## Phase 5: LIVE WIRING (user-approved 2026-08-31)

- [x] T014 Wire entry_gate into supervisor.py + paper_live.py; service env
      AUTOTRADE_ENTRY_WINDOW=1 (both services); restart + journalctl verify
      ("entry-window: 09:30-12:00 ET [ON]"); gate unit-checked (False at
      04:45 ET, fail-open on error); full suite 165/165 green

## Dependencies

Phase 1 -> 2 -> 4 sequential; Phase 3 parallel-safe with 2. No live changes
at any point.
