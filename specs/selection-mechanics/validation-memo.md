# Validation Memo: keep the roster and the winner-pick rule as-is

Date: 2026-09-01 | Source: specs/selection-mechanics/audit-report.md

## ONE recommendation: NO CHANGE — the selection mechanics are confirmed
by the full 5.3y audit. No lane removal, no winner-pick change, no follow-up
simulation.

## Why
1. The winner-pick rule ("highest proba wins") was fine-tuned by the data
   itself: ema and orb co-fire on the same bar only 6 times in 5 years
   (0.48% of trades), always in the same direction. The knob is immaterial —
   there is nothing to optimize. The dual-book simulation that would test
   alternatives is NOT warranted (the census was the gate, and the gate says
   stop).
2. The roster (ema on all 5, orb on NQ/ES/GC) is confirmed: all 8 active
   lanes positive on full history, 7 of 8 positive on the hold-out, and the
   only hold-out-negative lane (RTY ema, -0.07, n=20) fails the pre-registered
   n>=30 bar — no removal on thin evidence.
3. Zero removal flags means zero changes. The mechanism the user asked to
   fine-tune was already as good as the data can distinguish.

## What I will NOT do
Remove or add lanes based on this audit. Change the winner-pick rule.
Propose the dual-book simulation (the census closed it).

## Watch items (not actions)
- RTY ema on the hold-out (-0.07) and ES orb (weakest full-history lane,
  +0.37): as the live book accumulates trades under the 0.40-0.65 band,
  their hold-out samples grow; the removal rule can be re-run at any time
  (one command) and will flag them if they genuinely degrade.
- The 0.50-0.65 zone is untested territory (new under the user override);
  its outcomes accumulate live and will feed the next audit.

## Follow-ups available (user's call, none recommended now)
- Roster expansion (supertrend/keltner/bos/cisd_ote replays, ~hours) —
  no signal from this audit that it would add anything.
- Per-strategy exit shaping — different mechanism, would be its own cycle.
