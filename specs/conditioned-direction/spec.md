# Feature Specification: Regime-Conditioned Direction (the last direction question)

**Feature Branch**: `06-conditioned-direction`
**Created**: 2026-09-01
**Status**: Draft — awaiting user approval (gate)
**Input**: User description: "What is the right question for the bot to
predict chart direction?" — answered as: find measurable conditions where the
next-move direction beats a coin flip, out-of-sample, and where trading on
that condition improves the funnel.

## Background (why this experiment exists — and its honest prior)

Unconditional direction prediction is a measured dead end: 46-53% hit-rate
for every model family tried (YOLO, TTM, Chronos, fused, LLM-judgment),
70% would need signal correlation r≈0.59, best measured r≈0.27 (live) /
r≈0.19 (historical). BUT the doctrine also recorded the one place direction
showed a pulse: volatility/regime conditioning (r≈+0.39) and the r_hat
signal. This experiment asks the ONLY direction question the data hasn't
closed: **under which pre-registered, measurable conditions does the next
K-bar direction beat 0.50 — and does trading it improve the funnel?**
Honest prior, stated in advance: the most likely outcome is KILL or a weak
INCONCLUSIVE; if a condition survives, it is a genuine discovery — but the
question gets closed with numbers either way (the quality-band precedent).

## User Scenarios & Testing

### User Story 1 — Forward-direction dataset (Priority: P1)

For every funnel signal, compute the forward K-bar return
sign(close[ts+K] - close[ts]) from the bar CSVs (point-in-time: the forward
path is an outcome, never a feature). Horizons pre-registered: K = 10
(30 min) and K = 30 (90 min) bars — the bot's trade horizon.

**Why this priority**: nothing else runs without the outcome column.

**Independent Test**: forward returns reproduce the known bar path (spot-check
against the CSVs); monotonic in K; no look-ahead in features.

**Acceptance Scenarios**:
1. Given any signal, When the forward return is computed, Then it equals the
   close at ts+K minus close at ts from the same CSV.
2. Given the dataset, When built, Then every eligible signal has a forward
   direction (or is flagged "no data" at the tail).

---

### User Story 2 — Pre-registered conditions (Priority: P1)

FIXED condition list (chosen from the doctrine BEFORE looking at outcomes;
no data-peeking):
  1. EMA alignment: EMA10 above vs below EMA30 at ts (trend state)
  2. ADX state: ADX14 >= 18 vs < 18 (trend strength)
  3. Vol state: ATR14/ATR100 ratio >= 1.0 vs < 1.0 (chop vs flow — the
     regime-halt lever)
  4. RSI zone: RSI < 30, 30-70, > 70 (overbought/oversold)
  5. Hour-of-day: the 09:30-12:00 ET window vs outside
  6. Model r_hat: sign of r_hat (does the model's expected-R carry
     direction info?)
  7. Proba band: 0.40-0.50 vs 0.50-0.65 (the user-override split)
  8. Momentum: last 5-bar return >= 0 vs < 0

Condition hit-rate = fraction of signals under the condition whose forward
direction (K-bar) matches the condition's predicted direction (for
direction-neutral conditions like RSI zone, the prediction is
mean-reversion: RSI<30 predicts UP; for alignment/momentum: continuation).

**Why this priority**: the list is the pre-registration — the conditions are
chosen by theory, not by their results.

**Independent Test**: each condition's value is computed point-in-time
(trailing bars only) and cached per (symbol, ts) — reproducibility asserted.

**Acceptance Scenarios**:
1. Given the fixed list, When features are computed, Then every signal has
   all 8 condition values.
2. Given any condition value, When recomputed, Then it is identical
   (deterministic).

---

### User Story 3 — Robust selection + hold-out verdict (Priority: P1)

Selection (train folds, pre-2025-11-01, the 10 fold split from cycle 4):
a condition-horizon pair SURVIVES iff its hit-rate >= 0.53 in >=70% of folds
with n>=150 per fold (robust — no single lucky window). The best survivor is
the primary candidate. Hold-out (ts >= 2025-11-01, cycle-3 boundary):
  GO iff (a) bootstrap P(hit-rate > 0.50) > 0.95 (10k, seed 42),
  (b) n >= 150, (c) the conditioned book beats the baseline funnel on the
  hold-out (avgR AND PF, same four-bar structure), (d) August not worse.
KILL if nothing survives selection or the hold-out fails; INCONCLUSIVE if
thin. Secondary survivors reported, never deciding.

**Why this priority**: the multiple-testing guard — 8 conditions x 2 horizons
= 16 candidates make the train winner look good by luck; the hold-out alone
decides, and the book test (c) makes a hit-rate edge prove itself in R.

**Independent Test**: synthetic — an injected 55% condition with n>=150
passes selection and (if the book improves) GO; a coin-flip condition does
not survive; thin conditions do not.

**Acceptance Scenarios**:
1. Given a genuine 55% condition, When the rule runs, Then it survives
   selection.
2. Given the base rate (50%), When the rule runs, Then nothing survives
   (KILL — the direction question is then CLOSED with numbers).

---

### User Story 4 — Report + memo (Priority: P2)

`direction-report.md`: per-condition per-fold hit-rate tables (all 16
candidates, transparent), the survivor, and the hold-out numbers.
`validation-memo.md`: ONE recommendation — wire a direction overlay if GO /
close the direction question if KILL. Memo-gated, never auto-changed.

**Independent Test**: both files written with numbers.

**Acceptance Scenarios**:
1. Given the results, When the memo is written, Then one recommendation.

---

### Edge Cases

- Tail signals with no forward K bars: excluded from hit-rate (flagged).
- Direction-neutral conditions (RSI zones) use mean-reversion predictions —
  pre-registered in the list, not decided after seeing results.
- Condition value NaN (insufficient bars): treated as its own "unknown"
  bucket, reported, never silently merged.
- The 0.50-0.65 proba zone has few realized trades but MANY signals — the
  direction question uses SIGNALS (forward returns exist for every bar),
  so it is not limited by trade outcomes; the book test (c) uses trades.

## Requirements

### Functional Requirements

- **FR-001**: Forward returns from bar CSVs only; features point-in-time
  (trailing bars <= ts); no look-ahead.
- **FR-002**: Condition list, horizons (K=10,30) and hit-rate bar (0.53)
  fixed at spec time (US2) — no post-hoc condition selection.
- **FR-003**: Train folds = pre-2025-11-01 (cycle-4 fold split); hold-out =
  ts >= 2025-11-01 (cycle-3/4 continuity).
- **FR-004**: Verdict = four-bar rule (US3), bootstrap 10k seed 42,
  memo-gated.
- **FR-005**: No live config change during the experiment.
- **FR-006**: All 16 condition-horizon candidates reported; only the primary
  decides.

### Key Entities

- **DirectionCondition**: (condition, horizon, prediction rule, per-fold
  hit-rates, n).
- **ConditionedBook**: signals taken under the surviving condition vs
  baseline funnel (hold-out stats).

## Success Criteria

- **SC-001**: Forward-direction dataset built and spot-checked (US1).
- **SC-002**: Verdict emitted per FR-004 with all bars shown.
- **SC-003**: Report + memo written; all 16 candidates transparent.
- **SC-004**: Zero live changes (git diff verified).

## Assumptions

- Bar CSVs (data/*_3min.csv) are the source of truth for forward paths
  (they end 2026-08-25; signals after that lack forward returns and are
  excluded — the tail is small).
- The direction question is answerable with signals (not just trades), so
  the thin trade sample does not gate the hit-rate measurement; the BOOK
  test (c) gates whether any hit-rate edge is worth trading.
- If nothing survives: the direction question is CLOSED with numbers,
  permanently recorded in the doctrine (the user's original dream gets a
  definitive, measured tombstone).
