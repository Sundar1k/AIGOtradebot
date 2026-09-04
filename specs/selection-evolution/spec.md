# Feature Specification: Selection-Layer Evolution (walk-forward search)

**Feature Branch**: `04-selection-evolution`
**Created**: 2026-08-31
**Status**: Draft — awaiting user approval (gate)
**Input**: User description: "Have an agent try thousands of possibilities in
the market and trade whichever strategy fits." — scoped to the SAFE version:
search the selection layer on the 5.3y replay with walk-forward validation;
live money only ever sees a pre-registered winner.

## Background (why this experiment exists — and what it is NOT)

The user's idea — try many strategies live, trade whichever fits — is the
classic self-optimizing system. Done live it is permanent post-hoc tuning on
real money: with enough candidates, some will look great by luck (multiple
testing), evaluation is too slow (1.5 trades/day vs thousands of candidates),
and it chases the regime that just ended. The SAFE version of the same idea:
search thousands of configurations in the simulator, select robustly with
walk-forward folds, and let ONE survivor face a pre-registered out-of-sample
test before any live change. This experiment IS that safe version.

Scope honesty: the doctrine freezes the detection logic (bot.py
detect/grade/enter/exit), the 3-min timeframe, the models, and the PPO exit —
so the searchable space is the SELECTION layer only (the doctrine's own
thesis: selection is the edge). Candidates are configs of existing knobs:
  floor  ∈ {0.25, 0.30, 0.35, 0.40}          (currently 0.35)
  ceil   ∈ {0.50, 0.55, 0.60, 0.65}          (currently 0.50)
  chop   ∈ {0.8, 1.0, 1.2, 1.5, 2.0}         (currently 1.0)
  = 4 x 4 x 5 = 80 candidates (plus the live baseline). Hundreds, not
  thousands — the frozen loop caps the space on purpose. Searching detection
  logic would re-litigate a settled doctrine and multiply the overfit risk.

## User Scenarios & Testing

### User Story 1 — Fast candidate evaluator (Priority: P1)

Evaluate any selection config (floor, ceil, chop_max) on the existing
98,363-row scored-signal dataset in milliseconds (a filter + stats, no new
replay). Chop: cache the ATR14/ATR100 ratio per signal ONCE (bars ≤ ts,
reusing the regime-halt machinery) so all chop thresholds are free.

**Why this priority**: the search needs hundreds of cheap evaluations.

**Independent Test**: evaluator(0.35, 0.50, 1.0) reproduces the known live
baseline (+0.43R/PF 1.80 full-history, within tolerance) — the same harness
honesty check as cycles 1-3.

**Acceptance Scenarios**:
1. Given the dataset, When the live config is evaluated, Then stats match
   the known baseline (±0.02R, ±0.05 PF).
2. Given a candidate with floor 0.40, When evaluated, Then only proba>=0.40
   rows are taken (verified against a manual filter).

---

### User Story 2 — Walk-forward folds + robust selection (Priority: P1)

Split pre-2025-11-01 history into ~6-month folds. Evaluate every candidate on
every fold. Selection: the candidate with the best MEDIAN fold avgR among
those with PF>=1.8 in >=70% of folds (robust — no single lucky fold wins).
Top-3 reported; ONLY top-1 decides.

**Why this priority**: median-across-folds kills the single-lucky-window
artifact; the hold-out (below) is the real test.

**Independent Test**: fold splitter has no time overlap/leakage (asserted);
a synthetic candidate injected as lucky-in-one-fold does NOT win selection.

**Acceptance Scenarios**:
1. Given the fold boundaries, When checked, Then folds are contiguous,
   non-overlapping, and no signal appears in two folds.
2. Given a candidate with one excellent fold and poor others, When selection
   runs, Then it does not rank top (median, not mean of the lucky fold).

---

### User Story 3 — Hold-out verdict (Priority: P1)

The pre-registered decision, identical structure to cycle 3: top-1 candidate
vs the LIVE config on the SAME hold-out as cycle 3 (ts >= 2025-11-01 —
continuity). GO iff ALL of:
  (a) P(ΔavgR > 0) > 0.95  (bootstrap 10k, seed 42, cand vs live config),
  (b) PF_cand >= PF_live,
  (c) Aug 2026 avgR_cand > avgR_live  (regime must not worsen),
  (d) n_cand >= 150 on the hold-out.
KILL when decisive against; INCONCLUSIVE when thin. Candidates 2-3 reported
as secondary observations — NO decision power (prevented multiplicity).

**Why this priority**: the multiple-testing guard — hundreds of candidates
make the train winner look great by luck; the hold-out alone decides.

**Independent Test**: synthetic — a candidate with a real hold-out edge ->
GO; lucky-on-train-only -> KILL/INCONCLUSIVE; thin -> INCONCLUSIVE.

**Acceptance Scenarios**:
1. Given a clear hold-out edge, When the rule runs, Then verdict = GO with
   all four bars shown.
2. Given a train-lucky candidate, When the rule runs, Then it does not GO
   (hold-out is the only decision slice).

---

### User Story 4 — Robustness report + memo (Priority: P2/P3)

Per-fold table for the top-3 candidates (report only) and a one-page memo:
if GO — propose the new config for live wiring (entries + management as
today); if KILL — the live config stays. Memo proposes; user decides; never
auto-changes live config.

**Independent Test**: `specs/selection-evolution/validation-memo.md` with ONE
recommendation; fold tables in `selection_validator/results/evolution_folds.json`.

**Acceptance Scenarios**:
1. Given the verdict, When the memo is written, Then one recommendation with
   numbers.

---

### Edge Cases

- Tie in median fold avgR: pre-registered tiebreak — higher PF, then fewer
  extreme negative folds (max drawdown of fold avgR), then first in grid.
- Candidate identical to live config: excluded from selection (it IS the
  baseline).
- A fold with zero trades for a candidate: PF undefined -> counts as a failed
  fold (cannot satisfy the >=70% rule) — no candidate is penalized to KILL on
  one empty fold, but cannot win either.
- Chop ratio NaN (insufficient bars): treated as not-blocking (fail-open,
  live semantics).
- The 09:30-12:00 ET window gate: FIXED (validated cycle 3) — not searched.
  All evaluations run within the window filter, mirroring the live config.

## Requirements

### Functional Requirements

- **FR-001**: Evaluator MUST use the shared evaluate() path on the existing
  dataset; no new replay, no GPU (search = filters + stats).
- **FR-002**: Chop evaluation MUST cache one ATR ratio per signal (bars <= ts,
  trailing 300) and derive all thresholds from it — point-in-time.
- **FR-003**: Walk-forward folds MUST be contiguous, non-overlapping,
  covering pre-2025-11-01; hold-out = ts >= 2025-11-01 (cycle-3 boundary).
- **FR-004**: Selection = best median fold avgR among PF>=1.8 in >=70% of
  folds; top-1 only decides; top-3 reported.
- **FR-005**: Verdict MUST be numbers-backed (verdict.md), four bars shown,
  bootstrap 10k seed 42.
- **FR-006**: Live config/loop MUST NOT change during the experiment; memo
  proposes.
- **FR-007**: Window gate, detection logic, models, exit logic FROZEN — only
  the pre-registered grids (floor/ceil/chop) are searched.

### Key Entities

- **CandidateConfig**: (floor, ceil, chop_max) + fold stats.
- **EvolutionResult**: per-candidate per-fold table, selection ranking,
  hold-out verdict.

## Success Criteria

- **SC-001**: Evaluator reproduces the live baseline (harness honesty).
- **SC-002**: Verdict emitted per FR-005 with all four bars.
- **SC-003**: Robustness tables + memo written.
- **SC-004**: Zero live changes during the experiment (git diff verified).

## Assumptions

- The existing dataset + chop machinery suffice (no new data, no GPU).
- Hundreds of candidates is the right search scale (the frozen loop caps the
  space; thousands would be detection-logic search, which is off-limits).
- Hold-out continuity with cycle 3 (ts >= 2025-11-01) keeps the evidence
  comparable; the window filter applies to all evaluations.
