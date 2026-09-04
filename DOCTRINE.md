# DOCTRINE — Predict Less, Select Better

HARD RULE for every agent, session, and script touching this bot.
Codified 2026-08-21 after 5 years of the user's own data settled it.

## The principle
- PREDICT = asking "which way does the next bar go?" — measured
  impossible at every tradeable candle for price-only models.
- SELECT = asking "of the signals the engine fires, which ones do we
  take?" — this is where the edge lives, and it is measurable.

## The measured evidence (do not re-run, do not re-litigate)
- Predictability spectrum (predictability_spectrum.py, 5y, 5 symbols):
  15-min hit-rate 46.1% (z=-27, real REVERSAL = bid-ask bounce, costs
  eat it), 1h 48.0% (z=-7, same), 4h/daily coin flips, weekly 52.3%
  (z=+0.8, unproven, n=267). NO tradeable candle predicts direction
  profitably after costs.
- Winrate ceiling: 70% needs signal correlation r≈0.59; best measured
  r≈0.27 (TTM live) / r≈0.19 (historical). Every model family landed
  49-53%: YOLO, TTM zero-shot/fine-tuned (4 TFs, 3 depths), fused,
  confidence selection, 849-trade history, live RL loop (63 cycles).
- The bot's edge: raw engine ~29% WR at 2R/1R (LOSES); the selected
  subset 47.2% WR / +0.58R / PF 2.11 (PROFITS). Same signals, same
  market — the funnel is the edge.
- Timeframe: settled. 3-min stays. (timeframe_compare.py: raw engine
  loses at 3m/5m/15m/60m; longer candles just lose slower with 20x
  fewer trades.)

## Sanctioned winrate levers (the ONLY allowed ones)
1. Quality-band selection — gate entries to the top score band AFTER
   validation at N>=150 scored trades (AUTOTRADE_QUALITY_MIN).
2. Volatility/regime gating — skip extreme-vol / panic states
   (r=+0.39 vol signal is real; AUTOTRADE_VOL_GATE, regime gate).
3. Flow experiments — Phase 4 variants that beat the baseline
   out-of-sample (avgR AND PF, n>=100, bootstrap CI), max one winner
   per node.

## Forbidden
- Training models to predict direction "harder" (more epochs, bigger
  models, more features for direction) — measured dead end.
- Chasing 70% winrate as a target. Expectancy is the target.
- Changing the 3-min timeframe.
- Modifying the proven loop logic (bot.py detect/grade/enter/exit).

Every future session: read this file. The skill (topstep-trading-bot,
HARD DOCTRINE section) is the always-loaded copy of this rule.
