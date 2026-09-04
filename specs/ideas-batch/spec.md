# Feature Specification: Ideas Batch — four cheap selection filters

**Feature Branch**: `07-ideas-batch`
**Created**: 2026-09-01
**Status**: Draft — standing approval ("keep trying different ideas", 2026-09-01)
**Input**: Four untested, cheap, dataset-level selection ideas, each with a
pre-registered hold-out bar. Winner(s) go to a memo; losers are archived with
numbers. All use the existing dataset + bar CSVs — no replay, no GPU.

## The four ideas (pre-registered BEFORE looking at outcomes)

1. **GAP FILTER** — the overnight gap (today's open minus yesterday's close)
   is a classic conditioning variable. Two pre-registered variants:
   - gap-aligned-only: take only signals whose direction matches the gap
   - gap-opposed-only: take only signals against the gap
   (Both variants are tested; the literature is genuinely mixed, so the
   direction of any effect is not assumed.)

2. **JUMP FILTER** — the config's dormant factor-zoo lever (JUMP_ATR_MULT=0,
   never enabled): exclude signals whose bar (or one of the last 2) moved
   |close-close| > 1.5 x ATR(20) — theoretically unpredictable bars
   (Aleti/Bollerslev/Siggaard). Pre-registered mult = 1.5.

3. **VOLUME CONFIRMATION** — signals on rising volume (signal-bar volume >
   mean of prior 20 bars) vs falling volume. Volume is in the CSVs, never
   used by the funnel. Pre-registered: confirmed-only vs baseline.

4. **FIRST-HOUR NARROWING** — 09:30-10:30 ET only vs the full 09:30-12:00
   window. Cycle 3 measured the first hour as the stronger half (+0.64 vs
   +0.27 OOS) as a secondary observation; this is its formal pre-registered
   test with the full four-bar rule.

## The common pre-registered bar (per idea)

On the hold-out (ts >= 2025-11-01, cycle-3 boundary), with bootstrap 10k
seed 42:
  GO   iff P(ΔavgR > 0) > 0.95 AND PF_cand >= PF_base AND n_cand >= 150
       AND August avgR_cand > avgR_base
  KILL if decisive against; INCONCLUSIVE if thin.
The baseline is the CURRENT live funnel (window + floor 0.40 + ceil 0.65,
per the user-override config — the filters are tested ON TOP of it).
Memo-gated: winners are proposed for live wiring; losers archived.

## User Stories

- US1: build the per-signal feature caches (gap, jump, volume, hour) from
  the bar CSVs — point-in-time, deterministic, spot-checked.
- US2: per-idea candidate books vs baseline on train folds (measurement)
  and the hold-out (the only decision), with the common bar.
- US3: report (all ideas, all variants, numbers) + memo (ONE recommendation
  per winning idea; no winners = "none of the four add anything").

## Requirements

- FR-001: features point-in-time (trailing bars only); no look-ahead.
- FR-002: all variants pre-registered in this spec; no post-hoc additions.
- FR-003: hold-out = ts >= 2025-11-01; folds = cycle-4 split for reporting.
- FR-004: common four-part bar; bootstrap 10k seed 42; memo-gated.
- FR-005: no live config change during the experiment.

## Success Criteria

- SC-001: all four ideas measured on the hold-out with the common bar.
- SC-002: report + memo written; zero live changes (git diff verified).
