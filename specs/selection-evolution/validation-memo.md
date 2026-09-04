# Validation Memo: keep the live config; revisit after the hold-out grows

Date: 2026-08-31 | Source: specs/selection-evolution/verdict.md

## ONE recommendation: KEEP the live config (0.35 / 0.50 / chop 1.0) for now;
re-run the pre-registered verdict when the hold-out slice reaches n>=150
(~weeks of forward accumulation), then wire the candidate if it still holds.

## Why keep (the rule, not the vibes)
The search found a candidate (floor 0.40, ceil 0.65, chop 2.0) that beat the
live config on EVERY quality bar on the held-out slice: P(ΔavgR>0)=0.9954,
PF 1.82 vs 0.50, August -0.817 vs -1.216. But it had 140 hold-out trades,
10 short of the pre-registered 150 bar. The pre-registered rule — written
before the search ran — says no GO below 150. Wiring it now would be the
"close enough" move the whole process exists to prevent.

## Why it's likely to hold (the evidence FOR the candidate)
- Three independent signals converge: the top-3 candidates all chose the
  same shape (floor up, ceiling up, chop loose); the fold tables show the
  edge is not one lucky window; and cycle 2 independently found chop 1.0
  over-throttles (57% of signals blocked). The candidate is not a fluke of
  the search — it is the same lesson the data has been teaching.
- The window context explains the shift: inside 09:30-12:00 ET the
  higher-proba zone isn't toxic (that toxicity was all-day), so the ceiling
  can loosen; and chop 1.0's throttle is pointless inside a window that is
  itself a quality filter.

## What I will NOT do
Wire the candidate today. Change chop, floor, or ceiling ad hoc. Treat
INCONCLUSIVE as GO.

## The plan (pre-registered continuation)
1. Live keeps trading the current config; the selection_validator dataset
   keeps growing with each live/paper trade (already wired via the
   on_graded_signal hook).
2. When the hold-out slice (ts >= 2025-11-01) holds >=150 closed trades
   for the candidate, re-run `python -m selection_validator.evolve_search`
   — same grids, same rule, same boundary. The candidate remains
   out-of-sample (selected blind of everything >= 2025-11-01).
3. If the four bars then pass -> propose wiring (floor 0.40, ceil 0.65,
   chop 2.0) with the same paper-first caution as cycle 3. If not -> KILL
   and the search is archived like every other dead lever.
