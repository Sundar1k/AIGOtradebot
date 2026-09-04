# Audit Report: Selection Mechanics (cycle 5)

Generated: 2026-09-01 | Dataset: 98,363 rows, 11,364 funnel | Eligibility:
LIVE band (floor 0.40 / ceil 0.65) + 09:30-12:00 ET window + not jump-skipped
(FR-003) | Hold-out: ts >= 2025-11-01 (cycle-3 boundary) | Results:
selection_validator/results/mechanics_audit.json

## Part A — Dual-fire census: THE WINNER-PICK KNOB IS IMMATERIAL

| metric | value |
|---|---|
| dual-fire events (5.3 years) | **6** |
| fraction of eligible events | 0.48% |
| direction agreement | 100% |
| max-proba pick on dual-fires | n=6, WR 33%, avgR +0.21 |
| single-fire reference | n=1,241, WR 51.5%, avgR +0.82 |

ema and orb fire on the SAME bar essentially never (6 times in five years,
half a percent of trades, always agreeing on direction). The "highest proba
wins" competition — the step the user asked to fine-tune — almost never
happens. **No dual-book simulation is warranted**; the follow-up is closed
with numbers. (The small max-proba-pick sample shows nothing either way.)

## Part B — Active-lane audit: NO LANE QUALIFIES FOR REMOVAL

| symbol | lane | full n | full avgR | full PF | hold n | hold avgR | hold PF | symbol avgR |
|---|---|---|---|---|---|---|---|---|
| ES | ema | 125 | +0.71 | 2.78 | 15 | +0.65 | 3.04 | +0.51 |
| ES | orb | 187 | +0.37 | 1.82 | 21 | +0.40 | 2.16 | +0.51 |
| GC | ema | 71 | +0.71 | 2.98 | 13 | +1.00 | 3.04 | +0.70 |
| GC | orb | 208 | +0.69 | 2.90 | 41 | +0.25 | 1.50 | +0.70 |
| NQ | ema | 109 | +1.12 | 4.25 | 6 | +0.72 | 3.07 | +0.95 |
| NQ | orb | 229 | +0.86 | 3.86 | 32 | +0.42 | 2.36 | +0.95 |
| RTY | ema | 147 | +1.14 | 3.38 | 20 | -0.07 | 0.93 | +1.14 |
| YM | ema | 171 | +1.07 | 3.97 | 25 | +0.31 | 1.50 | +1.07 |

- Full history: ALL 8 lanes strongly positive (avgR +0.37 to +1.14, PF 1.8-4.3).
- Hold-out: 7 of 8 positive. RTY ema is the only negative hold-out lane
  (-0.07, n=20) — but the pre-registered removal rule requires n>=30, so it
  is NOT flagged (no removal on thin evidence — the rule's intent).
- **Removal flags: 0.** The roster (ema on all 5, orb on NQ/ES/GC) is
  CONFIRMED by the full 5.3y audit.

## Honest caveats

- Hold-out lane samples are small (n=6-41): the eligible band 0.40-0.65 has
  realized outcomes only for the 0.40-0.50 zone (the old config never traded
  0.50-0.65 — that zone is new under the user override and accumulates
  outcomes going forward). The removal rule's n>=30 bar exists exactly for
  this reason.
- ES orb remains the weakest lane (full +0.37/PF 1.82) — positive, not a
  removal candidate, but worth watching as outcomes accumulate.
- The audit measures the mechanism as it now runs (post user-override
  config); it does not revisit the override itself.

## Verdict summary

- Winner-pick fine-tuning: CLOSED (immaterial — 6 dual-fires in 5 years).
- Roster fine-tuning: CONFIRMED (0 removal candidates).
- Follow-ups proposed: none (the dual-book sim is not warranted; roster
  expansion remains available if ever desired, but the audit found no reason).
