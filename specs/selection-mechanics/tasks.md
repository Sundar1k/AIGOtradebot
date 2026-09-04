# Tasks: Selection-Mechanics Audit

**STATUS 2026-09-01: COMPLETE — NO CHANGE recommended. Census: 6 dual-fires in 5.3y (0.48%, 100% agreement) — winner-pick knob immaterial, dual-book sim closed. Lane audit: 0 removal flags (all lanes positive; RTY ema -0.07 @ n=20 < 30 bar). Roster + rule confirmed. T001-T013 done.**

**Input**: `specs/selection-mechanics/spec.md`, `specs/selection-mechanics/plan.md`
**Prerequisites**: plan.md + spec.md (both at approval gate)

## Phase 1: Census (US1)

- [x] T001 [US1] `selection_validator/mechanics_audit.py`: dual-fire events
      (same symbol+ts, both in-band at live floor/ceil 0.40/0.65,
      window-filtered, not jump-skipped) — count, fraction of taken trades,
      direction-agreement rate
- [x] T002 [US1] Chosen-pick performance on dual-fire events (WR/avgR/PF)
      vs single-fire book trades
- [x] T003 [P] [US1] Unit tests: census determinism; dual-fire definition
      (same ts+symbol, in-band); agreement-rate calculation

## Phase 2: Lane audit (US2)

- [x] T004 [US2] Per symbol x lane: full + hold-out (ts >= 2025-11-01)
      n/WR/avgR/PF vs symbol and book averages
- [x] T005 [US2] Flag hold-out-negative lanes (n>=30) as removal candidates
- [x] T006 [P] [US2] Unit tests: per-lane rows complete; hold-out boundary
      matches cycle 3

## Phase 3: Removal rule + report (US3, US4)

- [x] T007 [US3] Removal flag iff hold-out avgR < 0 AND n>=30 AND bootstrap
      P(avgR < symbol avgR) > 0.95 (10k, seed 42) — memo-gated, never
      auto-changed (FR-005)
- [x] T008 [P] [US3] Synthetic tests: clearly-negative lane -> flagged;
      weak-but-positive -> not flagged; zero hold-out trades -> not flagged
- [x] T009 [US4] `specs/selection-mechanics/audit-report.md` (census + tables
      + flags)
- [x] T010 [US4] `validation-memo.md` — ONE recommendation (remove flagged
      lane / keep roster / propose dual-book follow-up if census says the
      winner-pick knob matters)
- [x] T011 Commit per spec-kit rule
- [x] T012 Verify `git diff config.py supervisor.py bot.py` EMPTY (SC-004)
- [x] T013 Update tasks.md status + skill (algo-trading-bot-workflow)

## Dependencies

Phase 1 -> 2 -> 3 sequential; no live changes at any point (FR-005).
