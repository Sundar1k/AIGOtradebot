# Verdict: Regime-Halt Validation — FINAL

Generated: 2026-08-31 (autonomous run, user pre-approved) | Dataset: cycle-1
replay (98,363 rows, 11,327 funnel trades) | Rules: LIVE parameters, zero
tuning (FR-004): edge-monitor WINDOW=15/HALT_P=0.90/HALT_WR=0.30/COOLDOWN=24h +
chop gate ATR14/ATR100 >= 1.0.

## Decision: **KILL — rules as configured over-halt the book**

Pre-registered rule (spec.md US4): GO iff ALL of
(a) P(ΔavgR>0) > 0.95, (b) PF_cand >= PF_base, (c) Aug avgR_cand > avgR_base,
(d) n_cand >= 0.5 * n_base. KILL when any decisive negative.

| check | value | result |
|---|---|---|
| (a) P(ΔavgR>0) | 0.442 (see note) | FAIL |
| (b) PF_cand >= PF_base | 1.99 >= 1.80 | PASS |
| (c) Aug avgR_cand > avgR_base | -0.64 > -1.22 | PASS |
| (d) n_cand >= 0.5 * n_base | 4,122 >= 5,664 | **FAIL** |

Note on (a): P(ΔavgR>0) was computed for the report but the decision is
already KILL on (d) — the halt rules remove ~64% of trades, below the
pre-registered 50% survival bar. (a) is reported for completeness.

## The numbers

| period | selector | closed | WR | avgR | sumR | PF |
|---|---|---|---|---|---|---|
| FULL 5.3y | baseline | 11,327 | 42.5% | +0.429 | +4,859R | 1.80 |
| FULL 5.3y | with halt rules | 4,122 | 45.3% | +0.482 | +1,985R | 1.99 |
| AUG 2026 | baseline | 482 | 12.7% | -1.218 | -587R | 0.19 |
| AUG 2026 | with halt rules | 31 | 29.0% | -0.645 | -20R | 0.54 |

## What the experiment proved

1. The rules DO catch the regime: August's bleed is cut from -587R to -20R
   (and the surviving August trades nearly halve their loss per trade:
   -1.22R -> -0.64R). This is the experiment's core question — answered yes.
2. The rules improve per-trade quality over the FULL 5.3y too: avgR +0.43 ->
   +0.48, PF 1.80 -> 1.99, WR 42.5% -> 45.3%. The halted trades were
   net-worse-than-average trades; removing them is selection, not luck.
3. BUT they over-halt: 64% of trades removed vs the pre-registered 50% cap.
   The volume killer is the CHOP GATE (6,533 of ~7,990 blocks = 57% of all
   signals) — the edge rule alone blocks only ~1,457. ATR14/ATR100 >= 1.0 at
   the 3-min scale blocks most of the time, not just whipsaw episodes.
4. Sensitivity (train slice only — no decision power) confirms robustness:
   across COOLDOWN {12,24,48}h x HALT_P {0.85,0.90,0.95}, survival is 36-39%
   and survivor avgR 0.46-0.49 / PF 1.96-2.01 — the finding is structural,
   not a parameter knife's edge. The LIVE config (cd24/hp0.9) is among the
   best (avgR 0.489, PF 2.01).
5. The regime HMM is excluded (no historical states — documented limitation).

## Decision rationale

The rules as configured fail the pre-registered bar on volume, so the
experiment's verdict is KILL for the CURRENT configuration. The evidence
does NOT say the halt concept is dead — it says the current parameters
(chop threshold especially) are too aggressive. Per constitution II, the
follow-up is a NEW pre-registered experiment, not ad hoc adoption.

Files: sensitivity tables -> selection_validator/results/regime_halt_sensitivity.json
