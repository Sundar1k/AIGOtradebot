# VALUE-EDGE FAILURE — pre-registered experiment spec (2026-09-04)

Status: PRE-REGISTERED — written BEFORE any code. Verdict recorded only after the
pre-registered bars are met. Nothing here wires to live config.

## Source
Chris Creamer (Robbins World Cup champion, IQCapital interview, video PL7LKUsCgIQ,
transcript ~/reference/strategies/creamers-orderflow-robins-cup.txt). Feasibility
assessment by the fable profile 2026-09-04 (session 20260904_122939_c06c8e).

## Hypothesis
A price×volume-location structure — penetration of the prior session's value-area
edge into discount/premium, failure, then a SECOND attempt that fails higher/lower
(higher-low for longs) — carries selection edge for mean reversion back toward value
on 3-min RTH bars, within the 09:30-12:00 ET window. This is a NEW SELECTOR/source
(price×volume location), NOT a new direction predictor. Direction question stays closed.

## Data
- Bars: data/{SYM}_3min.csv (clean re-backfill, 0 sub-minute rows; 2021-04 -> 2026-08).
- Symbols: NQ, ES, RTY, YM, GC. 3-min bars. Times converted ONCE per symbol to
  America/New_York (DST pitfall — never fixed UTC offsets).
- RTH session = 09:30-16:00 ET. Prior-day VA = built from the previous RTH session's
  bars ONLY (point-in-time: yesterday's completed session).

## Parameters (fixed at pre-registration, no post-hoc tuning)
- VA: volume-by-price 70% value area around POC from prior RTH session.
  Profile bin = max(0.25, tick*4) price buckets; volume midpoint-attributed.
- Direction context (E1 proxy): prior RTH session close >= prior POC -> longs only;
  close <= prior POC -> shorts only; otherwise (inside/no clear) -> no trades that day.
- Penetration: price must trade at least PEN_ATR = 0.30 * prior-day ATR14 below VA_LOW
  (longs) / above VA_HIGH (shorts) before failure counts. PEN_ATR = 0.30.
- Failure: within FAIL_BARS = 12 bars of first penetration, a bar CLOSES back on the
  value side (>= VA_LOW for longs). L1 = extreme low of the penetration leg
  (min low from first touch until the failure close).
- Retry: after failure, a NEW leg trades back beyond VA_LOW (longs) but its low HOLDS
  above L1 (fails higher) within RETRY_BARS = 20 bars of the failure close.
  Retry low must be <= VA_LOW (the attempt actually reaches discount).
- Entry: close of the first bar whose close >= max(VA_LOW, retry-bar high) after a
  valid retry (the flip). Longs only when entry window 09:30-12:00 ET (signal bar ts).
- Stop: min(L1, retry low) - 0.10 * prior-day ATR14 (longs). Symmetric for shorts.
- Targets (both reported, pre-registered variants):
  A: POC of prior session (reversion to value center).
  B: prior RTH session high (longs) / low (shorts) (his swing target).
- Exit: stop-first on same-bar collisions (conservative, matches SimBroker). Session-end
  exit at 16:00 ET close if neither hit. No breakeven ratchet, no trail in v1.
- Participation filter: signal bar volume >= 0.5 * same-hour RTH median volume (prior
  30 sessions) — E9 proxy. Skip signal otherwise.
- Max 1 trade per direction per day; first valid signal wins.

## Pre-registered bars (spec-kit convention)
- OOS slice: ts >= 2025-11-01 (fixed, same boundary as funnel cycles).
- Sample: n_OOS >= 150 across symbols before any GO verdict (per doctrine; else
  INCONCLUSIVE).
- GO iff BOTH:
  1. bootstrap (10k, seed 42) P(meanR_OOS > 0) >= 0.95, AND
  2. meanR_OOS >= 0.30R AND PF_OOS >= 1.5 (mean-reversion edge at our ceiling; his
     60-65% WR is marketing, not a bar), AND
  3. combined-symbol OOS avgR beats the time-window funnel reference on the same
     slice (ref: +0.455R / PF 1.93, entries-gate config) — must ADD, not duplicate.
- KILL if n_OOS >= 150 and any bar fails. INCONCLUSIVE below n=150.
- Attribution check at GO time: dual-fire vs existing lanes on identical bar ts
  (census ~0.48% historically — confirm no duplication).

## Failure path (what "this fails" looks like, no loops)
If the selector cannot reach n=150 OOS in ~1,300 eligible RTH days at <=2 signals/day,
or its OOS stats sit at/below the funnel baseline, the experiment is KILLED and recorded.
No parameter grinding. No re-open of direction. One rerun allowed only for a
pre-registered code-fix (look-ahead bug), same bars.
