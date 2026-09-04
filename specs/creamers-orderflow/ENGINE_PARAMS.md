# CREAMER ORDERFLOW v2 — engine parameters (locked 2026-09-04, pre-run addendum)

Binding addendum to SPEC.md. Written BEFORE any tick data was run through the engine.
These turn the spec's qualitative trigger into deterministic gates derived from his
described mechanics (round numbers, no tuning against outcomes). Run identity: hermes
session bg_125245_fed2c0 (user: "complete this strategy, check live performance").
The v1 OHLCV engine (value_edge_failure.py, KILLED) is the baseline; v2 = same
mechanics + footprint/delta confirmation gates (the part v1 could not see).

## Timeframe decision (scope clarification, not tuning)
- Primary: 3-min footprint bars (identical bar grid to the funnel + v1 proxy, so the
  OOS comparison vs the window-funnel reference (+0.455R/PF 1.93, 3-min bars) is
  apples-to-apples and v2 is a controlled upgrade of v1).
- Secondary (reported, never deciding): 5-min footprint bars (his stated chart).
- GO/KILL pre-registered bars in SPEC.md apply to the PRIMARY 3-min run.

## Location/context gates (identical to v1)
- Prior RTH day (09:30-16:00 ET): volume profile from tick volume, bin width =
  max(tick*4, 0.25) index points (tick = 0.25 NQ-equivalent, 0.25 ES-equivalent);
  70% value area VA_low/VA_high + POC; ATR14 from prior-day bars.
- Direction filter: prior close > POC  -> longs eligible; prior close < POC -> shorts.
- Entry window: today 09:30-12:00 ET only. Penetration depth: 0.30 * prior ATR.
- Failure close: bar closes back beyond the VA edge (long: close >= VA_low).
- Retry: within RETRY_BARS=20 bars after failure, a bar re-crosses the edge but does
  NOT make a new extreme vs L1 (long: low > L1). Flip: close >= max(edge, retry_high).
- One trade per side per day (first completion wins).

## NEW footprint/delta confirmation gates (the v2 test)
All measured on the extreme bar = the bar that prints L1 (the deepest penetration bar).
Long setup (mirror for short, signs reversed):
- N1 CONCENTRATION: price levels in the bottom 25% of the L1 bar's range carry >= 25%
  of that bar's total tick volume (participation building at the extreme).
- N2 ONE-SIDED DELTA: L1-bar net delta (avol - bvol) <= -0.20 * L1-bar total volume
  (sellers aggressive in discount).
- N3 WEAKER RETRY: the bar that prints retry_low has net delta >= L1-bar delta
  (sellers fail higher with LESS aggression than the first push).
- N4 DOMINANCE SHIFT: the entry/flip bar (close >= max(edge, retry_high)) has
  net delta >= 0 (buyers lifting offers / delta flips positive).
- VOL_FLOOR (v1 rule, kept): signal-bar tick volume >= 0.5 * same-clock-hour median
  of prior 30 sessions.

## Simulation (identical to v1)
- Entry at flip-bar close; long stop = min(L1, retry_low) - 0.10*ATR (short mirror);
  target variants: A = prior POC, B = prior swing (long: prior high; short: prior low).
- Stop-first on collisions; exit at 16:00 ET close if neither hit. Costs implicit in
  the funnel reference comparison (same convention as v1: none added; risk-based R).

## Data (per SPEC.md) — resolved 2026-09-04, roster locked pre-run
- Feed resolved: Dukascopy JETTA public API (jetta.dukascopy.com/v1, no key), tick
  endpoint, codes USATECH.IDX-USD (NQ proxy), USA500.IDX-USD (ES proxy), USA30.IDX-USD
  (YM proxy), USSC2000.IDX-USD (RTY proxy) — ticks available since 2012-2013, covering
  the full window. Original SPEC.md names NQ=US100.IDX / ES=US500.IDX; the bot universe
  is NQ/ES/RTY/YM, so the index-proxy roster is the 4 CASH INDEX CFDs above. Gold is NOT
  included: GC is a commodity futures footprint, outside the tested strategy's market
  class (his instrument is MNQ equity-index futures); metal CFD ticks would add a fifth
  pipeline for zero class-coverage. OOS slice: ts >= 2025-11-01 fixed.
- Tick files: 1 file per UTC hour, JSON (timestamp ms epoch, multiplier, first
  ask/bid, times[] = ms inter-arrival deltas, asks[]/bids[] = int price deltas
  x multiplier, askVolumes[]/bidVolumes[]). Parse verified on 2026-08-20 13h.
- Failure path: if JETTA proves infeasible at scale (throttling, size, licensing),
  the experiment is recorded KILLED-ON-DATA with the reason — no silent swaps.
