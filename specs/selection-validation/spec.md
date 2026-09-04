# Feature Specification: Selection Validator (SELECTION-VALIDATION experiment)

**Feature Branch**: `01-selection-validator`
**Created**: 2026-08-30
**Status**: Draft — awaiting user approval (gate)
**Input**: User description: "Prove the selection edge beats baseline out-of-sample at p<0.05, else kill."

## User Scenarios & Testing

### User Story 1 — Scored-trade dataset, point-in-time (Priority: P1)

Assemble a complete, point-in-time record of every graded signal the engine has
ever produced (backtest replay + live + paper), with features, funnel decisions,
gate outcomes, and realized R when the trade closed.

**Why this priority**: Without the dataset nothing else runs; every later story
consumes it. It is also the first point-in-time leak-check point.

**Independent Test**: `selection_validator/dataset.py` produces a JSONL with ≥1,000
scored signals; each row has ts, symbol, proba, r_hat, direction, strategy,
floor, ceil, gate results, veto_quality, outcome_r (null until closed); a
leak-check script confirms no feature uses bars after the signal bar.

**Acceptance Scenarios**:
1. Given the 5y NQ/ES/RTY/YM/GC bar data, When backtest replay runs with signal
   logging enabled, Then every graded signal (TAKE and SKIP) is appended with
   proba, r_hat, floor/ceil and the gates that fired.
2. Given the live and paper ledgers, When ingestion runs, Then closed trades
   carry realized outcome_r and open ones carry null without blocking the row.
3. Given any row, When leak-check runs, Then it errors on any feature timestamp
   > signal-bar timestamp.

---

### User Story 2 — Baseline selector reproduces the known edge (Priority: P1)

The current funnel (proba floor 0.35, ceil 0.50, live gate stack: news blackout,
regime, chop, edge monitor, veto advisory) is the BASELINE. It must reproduce
the doctrine numbers (+0.58R / PF 2.11) on its historical window as a sanity
check that the harness is honest before any candidate is tested.

**Why this priority**: A harness that cannot reproduce the known baseline
cannot be trusted to judge a candidate.

**Independent Test**: `selectors.py::BaselineSelector` over the historical
(pre-2026-08-01) slice reports WR, avgR, sumR, PF; assert avgR within ±0.05R
and PF within ±0.15 of 0.58 / 2.11, else harness bug — fix before proceeding.

**Acceptance Scenarios**:
1. Given the historical slice, When BaselineSelector runs, Then WR≈47%, avgR
   +0.53…+0.63, PF 1.96…2.26 — otherwise the harness is flagged broken.
2. Given a signal at proba 0.34, When the baseline evaluates it, Then it is
   rejected (below floor) and still recorded in the dataset.

---

### User Story 3 — Candidate selector: quality-band gating (Priority: P2)

Candidate = baseline funnel PLUS a veto-quality band gate (doctrine's sanctioned
lever #1: quality-band selection). Pre-registered sweep: quality ≥ q for
q ∈ {5, 6, 7}; proba sub-bands {0.35-0.40, 0.40-0.45, 0.45-0.50} are measured,
not tuned. No other levers.

**Why this priority**: Quality gating is the one sanctioned selection lever not
yet enforced live (QUALITY_MIN=0 today, collect-only); this is its pre-registered
validation.

**Independent Test**: candidate stats over the SAME historical slice, one run
per q value, same code path as baseline (shared evaluator).

**Acceptance Scenarios**:
1. Given the historical slice, When CandidateSelector(q) runs for each q,
   Then per-q WR/avgR/PF/sumR tables are written to `selection_validator/results/`.
2. Given a signal with veto_quality 4 and q=6, When evaluated, Then rejected.

---

### User Story 4 — Verdict harness: bootstrap kill-or-go (Priority: P2)

Compare candidate vs baseline on the OUT-OF-SAMPLE slice only (trades with
entry ts ≥ 2026-08-01 — the Aug regime, never used to pick q). One-sided
bootstrap (10k draws, seed 42, reusing edge_monitor.bootstrap_p_lt): candidate
wins only if P(ΔavgR > 0) > 0.95 AND PF_candidate ≥ PF_baseline. Else KILL.

**Why this priority**: This is the experiment's spine — the pre-registered
decision rule the user approved. It must exist before any candidate is trusted.

**Independent Test**: unit tests with synthetic data — a known injected edge
must produce GO; pure noise must produce KILL; a 1-trade slice must produce
INCONCLUSIVE, not a crash.

**Acceptance Scenarios**:
1. Given synthetic data with a real edge, When verdict runs, Then verdict = GO.
2. Given synthetic noise, When verdict runs, Then verdict = KILL.
3. Given N<30 OOS trades, When verdict runs, Then verdict = INCONCLUSIVE with
   the count stated (pre-registered: no GO below N=150 on the live-only slice;
   backtest-replay OOS may reach the bar sooner and is primary evidence).

---

### User Story 5 — Forward integration: feed first, gate only after confirmation (Priority: P3)

Wire the validated selector into paper_live.py as a LOG-ONLY feed (env-gated
AUTOTRADE_SELECTOR=feed): every signal gets the selector's accept/reject logged
alongside the bot's own decision, never affecting entries. Promote to a real
gate only after forward N≥150 with the same stats hold.

**Why this priority**: Deploying a gate before forward confirmation repeats the
pattern-lane mistake; the feed gives forward OOS with zero risk.

**Independent Test**: paper daemon restarts with the env set; journal shows
`selector: ACCEPT/REJECT (q=…)` per signal; entries unchanged (same trade
count as a control run without the env).

**Acceptance Scenarios**:
1. Given AUTOTRADE_SELECTOR=feed set, When a signal fires on paper, Then the
   selector decision is logged and the entry path is byte-identical.
2. Given forward N≥150 with stats holding, When the GO-to-gate check runs, Then
   the promotion is proposed to the user — never auto-enabled.

---

### Edge Cases

- OOS slice too thin (live-only): report both backtest-replay OOS and live-only
  OOS separately; GO requires live-only N≥150, KILL is decisive from either.
- Bootstrap degenerate (all R identical): flagged INCONCLUSIVE, not a win.
- Missing veto_quality (0): candidate treats it as below every q (reject);
  baseline ignores it — divergence is expected and recorded.
- Feature leakage detected: dataset rebuild is MANDATORY before any verdict.
- Gate-stack divergence between replay and live (e.g. chop gate off in replay):
  documented per-slice in the verdict, never silently merged.

## Requirements

### Functional Requirements

- **FR-001**: System MUST record every graded signal point-in-time (append-only
  JSONL: ts, symbol, proba, r_hat, direction, strategy, floor, ceil, gate
  results, veto_quality, outcome_r).
- **FR-002**: Baseline and candidate MUST share one evaluator code path.
- **FR-003**: Bootstrap MUST be one-sided, 10,000 draws, fixed seed 42.
- **FR-004**: Verdict MUST be a numbers-backed report (`verdict.md`), with
  N, WR, avgR, PF, ΔavgR, p-value per slice.
- **FR-005**: The live loop (bot.py detect/grade/enter/exit) MUST NOT be
  modified during the experiment (doctrine: proven loop is frozen).
- **FR-006**: Candidate lever set MUST be exactly veto-quality ≥ q, q ∈ {5,6,7}.
- **FR-007**: OOS boundary MUST be entry ts ≥ 2026-08-01, fixed at spec time.

### Key Entities

- **ScoredSignal**: one graded signal (fields per FR-001; outcome_r null while open).
- **SelectorResult**: per-signal accept/reject for baseline or candidate (signal_id, selector, decision, reason).
- **VerdictReport**: per-slice stats + bootstrap p + GO/KILL/INCONCLUSIVE + numbers.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Dataset ≥1,000 scored signals, leak-checked, reproducible.
- **SC-002**: Baseline reproduces +0.58R / PF 2.11 ± tolerance on history.
- **SC-003**: Verdict on OOS slice at N≥150: GO (P(ΔavgR>0)>0.95 AND PF_cand ≥ PF_base) or KILL (else) — decision with numbers, recorded.
- **SC-004**: Zero changes to bot.py/supervisor.py entry logic during the experiment.

## Assumptions

- Backtest replay through the EXISTING handle_bar (bot.py) is the primary OOS
  evidence source (same code path as live, point-in-time); live+paper are the
  forward confirmation slice.
- At the current 12.4% clear rate the live-only slice grows slowly (~5-15
  trades/week); the experiment does NOT wait on it — replay OOS (Aug 2026
  window is already ~6 weeks of bars) is the decisive slice.
- veto_quality is available on historical rows via the veto capture ledger
  where recorded; rows without it are marked q=0 and handled per edge case.
- Existing infra is reused: edge_monitor.bootstrap_p_lt, evolve.winrate_by_quality,
  backtest.py trade CSV writer, pytest (9.1.1, tests/ exists).
