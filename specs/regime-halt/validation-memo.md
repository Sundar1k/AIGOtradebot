# Validation Memo: the LIVE halt rules (keep / adjust / remove)

Date: 2026-08-31 | Source: specs/regime-halt/verdict.md (pre-registered replay)

## ONE recommendation: ADJUST — keep both rules live, then run the
pre-registered chop-threshold experiment to restore volume.

## Why keep (the evidence FOR)
- The rules caught August: the wipeout slice drops from -587R to -20R, and
  per-trade loss nearly halves (-1.22R -> -0.64R). This is exactly what the
  rules exist for, and they did it.
- Over 5.3y the rules improve every quality metric: avgR +0.43 -> +0.48,
  PF 1.80 -> 1.99, WR 42.5% -> 45.3%. The trades they remove are the
  net-losing subset — they are a selection lever, not noise.
- Sensitivity across cooldown/HALT_P shows the effect is structural, and the
  live parameters (24h / 0.90) sit at the best end of the measured grid.

## Why adjust (the evidence AGAINST the current config)
- 64% of trades are removed vs the pre-registered 50% survival cap. The book
  starves: ~4,100 trades over 5.3y across 5 symbols (~160/mo/symbol) is thin
  and keeps the N>=150 validation problem alive.
- The volume killer is identifiable and measured: the chop gate (threshold
  1.0) blocks 57% of ALL signals — the 3-min ATR14/ATR100 >= 1.0 condition
  fires in most regimes, not just whipsaw. The edge rule alone removes only
  ~13% of signals and is NOT the problem.

## What NOT to do (constitution II)
- Do NOT retune the chop threshold ad hoc — the sensitivity grid measured
  cooldown/HALT_P only; the chop threshold needs its own pre-registered
  experiment (it is the single most consequential parameter in the stack).
- Do NOT remove the rules — the evidence says they add per-trade value and
  catch regimes; removing them would re-expose August-type bleed.
- Do NOT change anything live today. The live config already runs these
  rules; this memo is a proposal, and the follow-up experiment is the
  approval-gated path to any change.

## Proposed follow-up (one sentence, to be spec'd if approved)
Pre-registered chop-threshold sweep (max_ratio {1.0 live, 1.2, 1.5} x edge
window {15 live, 20}) on the same dataset, success bar: >=50% survival AND
P(ΔavgR>0)>0.95 AND August protection retained — else KILL the adjust idea.
