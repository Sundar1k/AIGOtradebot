# Feature Specification: Time-Window Gate (09:30-12:00 ET)

**Feature Branch**: `03-time-window`
**Created**: 2026-08-31
**Status**: Draft — awaiting user approval (gate)
**Input**: User description: "Experiment: bot only opens trades US 09:30-12:00 ET
(cash open through noon). Measurement showed +0.660R / PF 2.54 vs +0.502 / 2.01
all-day, and August -0.856 vs -1.218 — must now be validated blind."

## Background (why this experiment exists)

Hour-of-day analysis on the cycle-1 dataset (98,363 signals, 11,327 trades):
the funnel's edge is not uniform across the day. The 04:00-16:00 ET day block
is positive EVERY year 2021-2026 (+0.60 to +0.69), while the 00:00-04:00 ET
night block is weak-to-negative (ES nights PF 0.80; Monday nights -0.21). The
user-selected window 09:30-12:00 ET (cash open through noon) measures
+0.660R / PF 2.54 on history vs +0.502 / 2.01 all-day, and its first hour
(09:30-10:30 ET) is the single best hour of the day (+0.790R / PF 2.72,
August -0.18R). These are MEASUREMENTS made by looking at the data — the
window choice is a hypothesis to be validated out-of-sample, not a verdict.

## User Scenarios & Testing

### User Story 1 — Window gate evaluation (Priority: P1)

Evaluate the pre-registered window gate (US 09:30-12:00 ET) against the
all-day baseline on the held-out OOS slice (fixed boundary, set at spec
time): OOS = trades with ts >= 2025-11-01 (the last ~25% of the dataset by
time). The gate has ZERO free parameters — it is a pure time filter applied
to the existing funnel trades.

**Why this priority**: this is the experiment.

**Independent Test**: candidate = funnel trades with ET minute in
[09:30, 12:00); baseline = all funnel trades; both evaluated on the OOS
slice through the shared evaluate() path; bootstrap via harness.bootstrap_diff
(10k draws, seed 42).

**Acceptance Scenarios**:
1. Given the OOS slice, When candidate and baseline stats are computed, Then
   both use the identical evaluator and the same slice.
2. Given the ET minute computation, When a signal at 11:59:59 ET is checked,
   Then it is inside the window; at 12:00:00 ET, outside.

---

### User Story 2 — Pre-registered verdict (Priority: P1)

GO iff ALL of (on the OOS slice):
  (a) P(ΔavgR > 0) > 0.95  (bootstrap 10k, seed 42, window vs all-day),
  (b) PF_window >= PF_all-day,
  (c) August 2026 avgR_window > avgR_all-day  (the regime that must not worsen),
  (d) n_window_OOS >= 100  (enough trades to judge).
KILL when decisive against (P <= 0.5, PF worse, or August worse).
INCONCLUSIVE when directionally better but insignificant.

**Why this priority**: the pre-registered decision rule; identical structure
to cycles 1-2.

**Independent Test**: synthetic data — window clearly better on OOS -> GO;
window better overall but worse in August -> KILL (condition c); window thin
(<100) -> INCONCLUSIVE. Unit tests in tests/test_time_window.py.

**Acceptance Scenarios**:
1. Given a clear OOS edge, When the rule runs, Then verdict = GO with all
   four conditions shown.
2. Given August-worse evidence, When the rule runs, Then verdict = KILL
   (condition c failed).

---

### User Story 3 — Stability report (Priority: P2)

Per-year (2021-2026) and per-symbol window-vs-all-day tables — reported as
evidence of whether the window effect is durable or driven by one year /
symbol. NO decision power (the verdict is US2's).

**Why this priority**: a 6-year-consistent effect is much more trustworthy
than a single-slice one; this makes that visible without touching the rule.

**Independent Test**: table written to
`selection_validator/results/time_window_stability.json`.

**Acceptance Scenarios**:
1. Given the dataset, When the tables are written, Then every year/symbol
   has its own row with n, avgR, PF for both selectors.

---

### User Story 4 — Validation memo (Priority: P3)

One-page memo: if GO — propose wiring the window gate into the LIVE
supervisor (entries gated to 09:30-12:00 ET, management continues 24h);
if KILL — state why and that no change happens. The memo proposes; the user
decides; never auto-changes live config.

**Why this priority**: the deliverable is a keep/change proposal.

**Independent Test**: `specs/time-window/validation-memo.md` written with one
recommendation.

**Acceptance Scenarios**:
1. Given the verdict, When the memo is written, Then it states exactly one
   recommendation with the numbers.

---

### Edge Cases

- Signal exactly at 09:30:00 / 12:00:00 ET: half-open interval [09:30, 12:00).
- DST: ET offset is UTC-4 in summer / UTC-5 in winter — the dataset is
  summer-heavy; the live gate must use America/New_York wall time, not a
  fixed UTC offset (implementation note for the memo).
- Open positions crossing the window edge: management continues; only NEW
  entries are gated (live semantics).
- The first-hour sub-window (09:30-10:30 ET) is reported as a secondary
  observation — NO decision power (prevented multiplicity).

## Requirements

### Functional Requirements

- **FR-001**: Window filter MUST use ET minute-of-day (UTC-4 for the dataset's
  summer period); half-open [570, 720) minutes from midnight ET.
- **FR-002**: Baseline and candidate MUST use the shared evaluate() path.
- **FR-003**: Bootstrap MUST be one-sided, 10,000 draws, seed 42.
- **FR-004**: OOS boundary MUST be fixed at spec time: ts >= 2025-11-01 UTC.
- **FR-005**: Verdict MUST be numbers-backed in verdict.md; the first-hour
  sub-window is reported, never deciding.
- **FR-006**: Live config and loop MUST NOT change during the experiment
  (memo is a proposal).

### Key Entities

- **WindowGateResult**: window vs all-day stats per slice (train/OOS/August),
  bootstrap P, verdict.
- **StabilityReport**: per-year and per-symbol rows (no decision power).

## Success Criteria

- **SC-001**: Verdict emitted per FR-004/FR-005 with all four GO conditions
  shown explicitly.
- **SC-002**: Stability tables written (US3).
- **SC-003**: Memo with ONE recommendation (US4).
- **SC-004**: Zero changes to live config/loop (git diff check).

## Assumptions

- The existing dataset suffices (window gate is a pure time filter; no new
  replay, no GPU).
- OOS = last 25% of history (2025-11-01 -> 2026-08-25) is a fair blind slice:
  the window hypothesis was formed from the full-history look, and the
  per-year stability table (US3) guards against year-driven artifacts.
- The user's intent is entries-only gating; exits/manage continue 24h.
