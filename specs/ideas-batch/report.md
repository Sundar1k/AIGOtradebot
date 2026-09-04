# Report: Ideas Batch (cycle 7)

Generated: 2026-09-01 | Dataset: 98,363 rows | Baseline = the CURRENT live
funnel (09:30-12:00 window + floor 0.40 + ceil 0.65, per user override) |
Hold-out: ts >= 2025-11-01 | Common bar: P(ΔavgR>0)>0.95 (10k seed 42),
PF_cand>=PF_base, n>=150, August not worse | Results:
selection_validator/results/ideas_batch.json

## The verdicts

| idea | hold_n | hold avgR | hold PF | P(delta>0) | Aug avgR | verdict |
|---|---|---|---|---|---|---|
| gap-aligned only | 106 | +0.367 | 1.81 | 0.479 | -0.64 | KILL |
| gap-opposed only | 67 | +0.397 | 1.71 | 0.527 | -0.99 | KILL |
| jump-excluded (1.5x ATR20) | 110 | +0.398 | 1.93 | 0.531 | -1.13 | KILL |
| volume-confirmed only | 96 | +0.294 | 1.59 | 0.349 | +0.07 | KILL |
| **first-hour only (09:30-10:30)** | **89** | **+0.404** | **1.92** | **0.554** | **+0.51** | **INCONCLUSIVE** |

## What each finding means

1. **Gap filter: dead (both directions).** Neither gap-aligned nor
   gap-opposed filtering improves the book (P=0.48/0.53 — no signal, PF at
   baseline). The overnight gap carries no tradeable information on top of
   the funnel. Closed.

2. **Jump filter: dead.** Excluding signals on/after a 1.5x-ATR jump bar does
   not improve the book (P=0.53). The factor-zoo idea is not additive here —
   the funnel's other filters already handle it. The dormant JUMP_ATR_MULT
   knob can stay dormant, now with numbers.

3. **Volume confirmation: dead.** Volume-confirmed signals are not better
   (P=0.35 — the weakest result; even the direction is unfavorable). Volume
   as measured adds nothing. Closed.

4. **First-hour narrowing (09:30-10:30 only): the one with a pulse —
   INCONCLUSIVE.** Directionally better (avgR +0.404 vs baseline, PF 1.92)
   and REMARKABLY August-resilient (+0.51 avgR in August, vs -1.22 for the
   full window — the only thing in this entire project that made money in
   August). But n=89 < 150 and P=0.554 — not significant, per the rule:
   INCONCLUSIVE, not GO. This matches cycle 3's secondary measurement
   (first hour +0.644 OOS, nearly flat in August). The August number is
   tantalizing but a single month — the rule exists exactly for this.

## The honest bottom line

Three of the four ideas are closed with numbers (gap, jump, volume — none
add anything). The fourth — narrowing to the first hour — has the strongest
August resilience measured anywhere in this project, but its sample is too
thin to trust (89 trades). The full window stays as-is; the first-hour
narrowing becomes the next re-test when its hold-out sample reaches 150
(live accumulation, same machinery, one command).
