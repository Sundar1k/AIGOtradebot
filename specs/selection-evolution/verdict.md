# Verdict: Selection-Layer Evolution — FINAL

Generated: 2026-08-31 (autonomous run, user pre-approved) | Dataset: cycle-1
replay (98,363 rows) | Search: floor {0.25-0.40} x ceil {0.50-0.65} x chop
{0.8-2.0} = 79 candidates + live config | Folds: 10 x ~6-month walk-forward
slices before 2025-11-01 | Hold-out: ts >= 2025-11-01 (cycle-3 boundary),
never touched by selection | Window gate 09:30-12:00 ET applied to ALL
evaluations (frozen, cycle-3 validated).

## Decision: **INCONCLUSIVE — the candidate is better on every quality bar
but 10 trades short of the pre-registered sample bar (140 < 150)**

Harness sanity (SC-001): window(0.35, 0.50, chop-off) = avgR +0.645 / PF 2.51
— reproduces cycle-3's full-history window numbers. The ruler is honest.

## The search

79 candidates x 10 folds evaluated (filters + stats on the dataset, no
replay, no GPU). Robust selection (FR-004): best median fold avgR among
PF>=1.8 in >=70% of folds.

Top-3 (robust):
  1. (floor 0.40, ceil 0.65, chop 2.0)
  2. (floor 0.40, ceil 0.60, chop 2.0)
  3. (floor 0.40, ceil 0.55, chop 2.0)

The pattern across all top candidates: raise the floor to 0.40 (tighter
quality band), loosen the ceiling to 0.55-0.65, loosen chop to 2.0. Inside
the 09:30-12:00 window, the model's higher-proba zone is NOT toxic (the
all-day toxicity was an outside-the-window artifact), and chop 1.0 over-throttles
(consistent with cycle 2's finding).

## The hold-out numbers (the only numbers that decide)

| config | trades | WR | avgR | sumR | PF |
|---|---|---|---|---|---|
| live (0.35, 0.50, chop 1.0) | 39 | 28.2% | -0.547 | -21.3 | 0.50 |
| candidate (0.40, 0.65, chop 2.0) | 140 | 42.9% | +0.423 | +59.2 | 1.82 |

Pre-registered four bars:
  (a) P(ΔavgR>0) > 0.95:  **0.9954**  PASS
  (b) PF_cand >= PF_live: 1.82 >= 0.50  PASS
  (c) Aug avgR_cand > avgR_live: -0.817 > -1.216  PASS
  (d) n_cand >= 150: **140 >= 150  FAIL**

## Why INCONCLUSIVE is the correct answer (and the discipline works)

The candidate is directionally strongly better (P=0.995, PF 3.6x, August
better) — but the pre-registered rule says no GO below 150 hold-out trades,
and the rule exists precisely for moments like this, when "close enough"
is the temptation. The verdict is INCONCLUSIVE, not GO and not KILL:
- not GO: n=140 < 150 (the bar is the bar);
- not KILL: every quality bar passed (P>0.95, PF, August).

## Path forward (pre-registered continuation, not post-hoc)

The hold-out slice grows as live/paper trades accumulate (same boundary
2025-11-01 — the candidate was selected blind of all of it, so any future
data remains out-of-sample for it). Re-running this verdict on the extended
dataset is cheap and is the legitimate continuation. Meanwhile the live
config stays exactly as it is.

Note for the memo: cycles 2 and 4 independently converge on the same
finding — chop 1.0 over-throttles the windowed book (39 hold-out trades vs
140 at chop 2.0). The chop-threshold question is the recurring lever.

Files: fold tables -> selection_validator/results/evolution_folds.json
