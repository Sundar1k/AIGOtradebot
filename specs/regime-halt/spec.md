# Feature Specification: Regime-Halt Validation

**Feature Branch**: `02-regime-halt`
**Created**: 2026-08-31
**Status**: Draft — awaiting user approval (gate)
**Input**: User description: "Validate the existing halt rules (edge-monitor realized-edge halt + chop gate) point-in-time over 5.3y — do they avoid August-type wipeouts without sacrificing the good months?"

## Background (why this experiment exists)

The selection-validation cycle (2026-08-30) proved the funnel edge is real but
regime-dependent: 5.3y baseline +0.50R / PF 2.01 (train), yet August 2026
collapsed to WR 12.7%, avgR -1.22R, PF 0.19 with EVERY symbol/strategy
negative. The doctrine's only real signal is regime (vol corr r≈+0.39). Two
deterministic halt rules already run live with FIXED parameters:
  - edge_monitor.py (enforce): halt entries while the trailing realized edge
    is statistically gone (P(meanR<0) > 0.90, or WR<0.30 with meanR<0, window
    N=15, 24h cooldown, then resume-to-test).
  - chop_gate.py: block signals with ATR14/ATR100 >= 1.0 (whipsaw).
This experiment replays BOTH point-in-time over the full scored-signal
dataset to answer one question: would they have avoided August while keeping
the good months? The rule parameters are NOT tuned — they are the live ones.

## User Scenarios & Testing

### User Story 1 — Point-in-time halt simulator (Priority: P1)

Simulate the two live halt rules over the existing scored-signal dataset
(98,363 rows, 11,327 trades — NO new replay, NO GPU). Book-level (the live
edge monitor is shared across all symbols): process signals chronologically
by ts across symbols; maintain the trailing window of closed trades; apply
the edge rule + chop gate at each signal; record GO/HALT per signal.

**Why this priority**: Without the simulator nothing else runs.

**Independent Test**: on synthetic data, a known losing streak trips HALT,
the 24h cooldown elapses, recovery trades resume, and cold start (window <
15 trades) takes everything — unit tests in `tests/test_regime_halt.py`.

**Acceptance Scenarios**:
1. Given 20 consecutive -1R trades, When the rule evaluates the 21st signal,
   Then it is HALTed (P(meanR<0) ≈ 1.0).
2. Given a halt at time T, When a signal arrives at T+2h, Then it is HALTed
   (cooldown); at T+26h, Then it is evaluated normally.
3. Given a signal with ATR14/ATR100 = 1.4 at its bar, When the chop gate
   runs, Then it is HALTed (chop).
4. Given fewer than 15 closed trades in the book, When any signal arrives,
   Then it is TAKEN (cold start — mirrors live).

---

### User Story 2 — Comparison: baseline vs halted funnel (Priority: P1)

Candidate = funnel trades that survive the halt rules; baseline = all funnel
trades (existing dataset). Compare on the FULL 5.3y replay (primary — the
rules never saw this data) and the August 2026 slice specifically (secondary
— the wipeout that must be avoided).

**Why this priority**: This is the experiment's question.

**Independent Test**: stats (n, WR, avgR, sumR, PF, max drawdown of sumR)
computed through the shared evaluate() path; bootstrap identical to the
previous cycle (10k draws, seed 42).

**Acceptance Scenarios**:
1. Given the full dataset, When candidate and baseline stats are computed,
   Then both use the identical evaluator (FR-002).
2. Given the August slice, When the same comparison runs, Then its stats are
   reported separately (the mandatory GO condition).

---

### User Story 3 — Sensitivity measurement, train slice only (Priority: P2)

Measurement (NO decision power): COOLDOWN 12/24/48h and HALT_P
0.85/0.90/0.95 variants, computed on the pre-2026-08-01 slice only. Reports
how sensitive the rule is; the verdict uses ONLY the live parameters.

**Why this priority**: Shows the rule isn't a knife's-edge artifact, without
letting the variants leak into the decision (constitution II).

**Independent Test**: table of (variant → n, avgR, PF, halt-fraction) written
to `selection_validator/results/regime_halt_sensitivity.json`.

**Acceptance Scenarios**:
1. Given any variant, When run on the train slice, Then its numbers are
   recorded but never enter the GO/KILL rule.

---

### User Story 4 — Verdict (Priority: P2)

Pre-registered rule, same machinery as cycle 1. Verdict on the FULL replay:
  GO iff ALL of:
    (a) P(ΔavgR > 0) > 0.95  (bootstrap 10k, seed 42, candidate vs baseline),
    (b) PF_candidate >= PF_baseline,
    (c) August slice avgR_candidate > avgR_baseline   ← the point of the rule,
    (d) n_candidate >= 0.5 * n_baseline               (didn't halt everything).
  KILL when decisive against (P <= 0.5, or PF worse, or August not improved).
  INCONCLUSIVE when 30 <= n < 150 directionally-better-but-insignificant.

**Why this priority**: (c) is mandatory — a rule that helps overall but
misses the very wipeout it exists for is a kill. No thin GO (n>=150).

**Independent Test**: synthetic data — losing streak → GO (avoided); no
streak → INCONCLUSIVE; rule halts everything → KILL (n_cand < 0.5 n_base).

**Acceptance Scenarios**:
1. Given a dataset with a clear wiped-out window, When the rule catches it,
   Then verdict = GO with all four bars shown.
2. Given a dataset where the rule helps overall but not in the wipeout
   window, Then verdict = KILL (condition (c) failed) — tested explicitly.

---

### User Story 5 — Validation memo for the live config (Priority: P3)

One-page memo: what the replay proved about the CURRENT live halt rules
(which are already running) — keep as-is (GO), consider adjusting, or remove
(only if KILL shows they cost more than they save). The memo proposes; the
user decides. Never auto-changes live config.

**Why this priority**: The rules are already live; the experiment's output is
a keep/adjust/remove recommendation, not new wiring.

**Independent Test**: `specs/regime-halt/validation-memo.md` written with the
verdict numbers and a single explicit recommendation.

**Acceptance Scenarios**:
1. Given the verdict, When the memo is written, Then it states one
   recommendation with the numbers that justify it.

---

### Edge Cases

- Cold start (<15 closed trades): taken — mirrors live warmup.
- Cooldown across symbol boundaries: book-level clock (real signal timestamps).
- Chop gate warmup (MIN_BARS=120): all dataset signals satisfy it (replay
  starts at bar 500); nan → fail-open (never blocks).
- Resume-to-test: after 24h cooldown the window is fresh; if still losing it
  re-halts — simulate the edge monitor's exact state machine.
- Regime HMM: EXCLUDED — only the current snapshot exists (no historical
  regime states); documented as a limitation, not a hole to fill with a proxy.

## Requirements

### Functional Requirements

- **FR-001**: Halt decisions MUST be point-in-time: only closed trades with
  ts < signal ts enter the trailing window; no look-ahead (constitution IV).
- **FR-002**: Both baseline and candidate MUST use the shared evaluate()
  path (selection_validator.selectors).
- **FR-003**: Bootstrap MUST be one-sided, 10,000 draws, seed 42 (reuse
  harness.bootstrap_diff).
- **FR-004**: Rule parameters MUST be the LIVE values (edge monitor
  WINDOW=15, HALT_P=0.90, HALT_WR=0.30, COOLDOWN_H=24, CLEAR_* tier-1 unused;
  chop gate threshold 1.0, MIN_BARS=120) — zero tuning.
- **FR-005**: Verdict MUST be numbers-backed in verdict.md with n, WR, avgR,
  PF, ΔavgR, P, halt-fraction, per period (full + August).
- **FR-006**: Live loop and config MUST NOT change during the experiment
  (doctrine: frozen; the memo is a proposal, not an edit).
- **FR-007**: OOS = the full 5.3y replay (the rules' parameters were set from
  a 281-trade ledger + live defaults, never from this dataset) with the
  August 2026 slice reported separately.

### Key Entities

- **HaltDecision**: per-signal GO/HALT + which rule(s) fired + trailing
  window stats at decision time.
- **RegimeHaltResult**: candidate vs baseline stats per period + halt
  fraction + bootstrap P + verdict.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Simulator reproduces the live edge-monitor state machine
  (normal/watch/halt/cooldown-resume) with unit-tested transitions.
- **SC-002**: Verdict emitted per FR-007/FR-005 with all four GO conditions
  shown explicitly.
- **SC-003**: Validation memo written with ONE recommendation.
- **SC-004**: Zero changes to live config or loop (verified by git diff of
  config.py / supervisor.py / bot.py at the end).

## Assumptions

- The existing scored-signal dataset is sufficient (it contains every funnel
  trade's outcome and every signal bar's ts); no new replay or GPU pass.
- Book-level halt (one shared edge monitor across symbols) matches the live
  supervisor wiring.
- The chop gate's ATR ratio is computable for every signal bar from the data
  CSVs (same reconstruction as the quality pass: trailing 300 bars ≤ ts).
- Historical regime-HMM states are unavailable; the regime gate is out of
  scope for this cycle and documented.
