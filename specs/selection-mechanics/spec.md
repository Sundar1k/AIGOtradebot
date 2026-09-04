# Feature Specification: Selection-Mechanics Audit (dual-fire + lane audit)

**Feature Branch**: `05-selection-mechanics`
**Created**: 2026-09-01
**Status**: Draft — awaiting user approval (gate)
**Input**: User description: "Fine-tune the per-bar strategy competition — how
the bot picks the winner and who competes per symbol." Scoped honestly: this
cycle MEASURES the mechanics on the existing dataset (minutes); any change
that requires new replays (dual-book simulation, adding lanes) becomes a
follow-up cycle only if the measurement shows the knob matters.

## Background (why this experiment exists)

The live mechanism: fixed roster (ema on all 5, orb on NQ/ES/GC only) ->
mechanical detection -> model grading -> highest-proba survivor in the band
-> gates -> PPO exit. Two fine-tunable parts exist:
  A) the WINNER-PICK rule (currently "max proba" — alternatives: max r_hat,
     proba x r_hat, strategy priority), and
  B) the ROSTER (orb OFF on YM/RTY per an Apr-Jun screen; ES ema is the
     weakest lane at +0.22R).
Neither has been audited on the full 5.3y dataset. This cycle audits both.
Honest scoping: the dataset contains outcomes ONLY for the lanes that were
ACTIVE (ema + orb on NQ/ES/GC) — so "would orb have worked on YM?" cannot be
answered from the dataset (no YM-orb trades were ever taken); it needs a
replay with an expanded roster. Same for the winner-pick rule: the
alternative's outcome on a dual-fire bar was never realized, so a full
comparison needs a dual-book simulation. Both are follow-up cycles, gated on
the findings here.

## User Scenarios & Testing

### User Story 1 — Dual-fire census (Priority: P1)

Measure how often two strategies fire on the same bar (same symbol, same ts,
both take-eligible in-band): frequency, direction agreement rate, and the
performance of the current max-proba pick on those events vs the single-fire
book. If dual-fires are rare or usually agree in direction, the winner-pick
knob is nearly irrelevant — a decisive honest finding that closes the topic
without an expensive simulation.

**Why this priority**: tells us whether Part-A fine-tuning is worth ANY
further work (dual-book replay is ~hours; the census is minutes).

**Independent Test**: census numbers reproducible (deterministic), and the
dual-fire set is explicitly bounded (count, symbols, date range).

**Acceptance Scenarios**:
1. Given the dataset, When the census runs, Then it reports dual-fire count,
   fraction of all taken trades, direction-agreement rate, and the chosen
   pick's WR/avgR vs single-fire trades.
2. Given a dual-fire rate below 5% or agreement above 90%, When the report
   is written, Then it states the winner-pick knob is immaterial (no
   follow-up simulation proposed).

---

### User Story 2 — Active-lane audit (Priority: P1)

Per symbol x lane (ema on NQ/ES/RTY/YM/GC, orb on NQ/ES/GC): full-history and
hold-out (ts >= 2025-11-01, cycle-3 boundary) stats — n, WR, avgR, PF — vs
the symbol's own average and the book average. Flags lanes that are
hold-out-negative or hold-out-weak relative to their symbol.

**Why this priority**: the roster was set by an Apr-Jun 2026 screen; the full
5.3y audit either confirms it (numbers) or surfaces a removal candidate.

**Independent Test**: per-lane stats reproducible; hold-out boundary matches
cycle 3.

**Acceptance Scenarios**:
1. Given the dataset, When the audit runs, Then every active lane has a row:
   full + hold-out n/WR/avgR/PF vs symbol average.
2. Given a lane with hold-out avgR < 0, When the audit runs, Then it is
   flagged as a removal candidate with its n stated.

---

### User Story 3 — Pre-registered decision (Priority: P1)

Removal rule (the ONLY change this cycle can trigger): a lane is a REMOVAL
candidate iff its hold-out avgR < 0 AND n >= 30 AND bootstrap
P(its avgR < its symbol's avgR) > 0.95. The decision itself is a memo
(propose removal; user decides) — never auto-changed. If no lane qualifies,
the roster is confirmed as-is with numbers. Dual-fire census findings are
reported; no winner-pick change is made this cycle (a dual-book simulation
would be a separate pre-registered cycle, only if the census shows the knob
matters).

**Why this priority**: one change type, one rule, memo-gated.

**Independent Test**: synthetic — a clearly-negative lane triggers the flag;
a weak-but-not-negative lane does not.

**Acceptance Scenarios**:
1. Given a lane with hold-out avgR -0.5, n=60, P<0.05 vs symbol, When the
   rule runs, Then it is a removal candidate.
2. Given a lane with hold-out avgR +0.1, When the rule runs, Then it is not.

---

### User Story 4 — Audit report + memo (Priority: P2)

`audit-report.md`: census + per-lane tables + flags. `validation-memo.md`:
ONE recommendation (remove flagged lane / keep roster / propose follow-up
dual-book cycle if the census says the knob matters). Never auto-changes live.

**Independent Test**: both files written with numbers.

**Acceptance Scenarios**:
1. Given the results, When the memo is written, Then one recommendation.

---

### Edge Cases

- Lane with zero trades on the hold-out: not evaluable -> no flag, reported
  as "insufficient data" (no removal on absence of evidence).
- Dual-fire definition: same (symbol, ts), both strategies' signals
  take-eligible (in-band at the CURRENT live floor/ceil 0.40-0.65) AND not
  jump-skipped.
- The current floor/ceil (0.40/0.65, user-override) applies to eligibility —
  the audit reflects the LIVE mechanism, not the pre-override one.
- Removal candidates: memo-gated; the live config is never changed by this
  cycle (the recent user override stands until the user says otherwise).

## Requirements

### Functional Requirements

- **FR-001**: Dataset-only analysis (no replay, no GPU) — reuse
  selection_validator.load_rows/evaluate.
- **FR-002**: Hold-out boundary = ts >= 2025-11-01 (cycle-3 continuity).
- **FR-003**: Eligibility = in-band at the LIVE floor/ceil (0.40/0.65) and
  window-filtered, mirroring the live mechanism.
- **FR-004**: Removal rule per US3 (hold-out avgR < 0, n>=30,
  P(avgR < symbol avgR) > 0.95, bootstrap 10k seed 42); memo-gated.
- **FR-005**: No live config change during the experiment; memo proposes.
- **FR-006**: Dual-fire census reported; no winner-pick change this cycle
  (follow-up cycle gated on the census).

### Key Entities

- **DualFireStats**: count, fraction, agreement rate, chosen-pick performance.
- **LaneAuditRow**: symbol, lane, full/hold-out n/WR/avgR/PF, symbol avg.
- **AuditVerdict**: flags + memo recommendation.

## Success Criteria

- **SC-001**: Census + audit tables produced (reproducible).
- **SC-002**: Removal candidates flagged per FR-004 with numbers.
- **SC-003**: Memo with ONE recommendation.
- **SC-004**: Zero live changes (git diff verified).

## Assumptions

- The existing dataset (98,363 rows) suffices — active lanes' outcomes are
  all present.
- Follow-up cycles (dual-book simulation, roster-expansion replays) are
  explicitly NOT this cycle; they are gated on the census/audit findings.
- The user-override config (0.40/0.65/chop 2.0) is the LIVE mechanism the
  audit mirrors; this cycle does not revisit it.
