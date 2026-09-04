# CREAMER ORDERFLOW (real footprint/delta) — pre-registered spec v2 (2026-09-04)

Status: PRE-REGISTERED before data pipeline build. Supersedes the v1 proxy experiment
(specs/value-edge-failure/ — that tested an OHLCV translation, KILLED; it is NOT a test of
Creamer's strategy and is recorded as such). This experiment tests HIS strategy's testable
core with real bid/ask order-flow data.

## What is being tested (his words, transcript ~/reference/strategies/creamers-orderflow-robins-cup.txt)
- Context/location: price in DISCOUNT/PREMIUM relative to session value area; first ~1.5h NY open.
- Confirmation (the part v1 could not see): footprint candles — concentration of volume at the
  candle extreme with one-sided DELTA (sellers aggressive in discount, unrewarded = absorption);
  candle flips back bullish.
- Trigger: first seller push fails; retry attempt FAILS HIGHER (higher low); flip -> long
  (mirror for shorts). Stop beyond the failed sellers' extreme. Target value area/POC or swing.
- He trades MNQ 5-min footprint. We test on footprint built from tick data of the equivalent
  CASH INDEX (Dukascopy free feed), 3-min and 5-min bars, NQ=US100.IDX / ES=US500.IDX.

## Data
- Source: Dukascopy public tick feed (no signup/key): datafeed.dukascopy.com, bi5 minute files.
- Proxy caveat (recorded, NOT hidden): cash-index CFD ticks, not CME futures. Aggressor-side
  delta is real bid/ask tick attribution; correlation with the futures footprint is high but
  not identical; queue position/perfect fills still assumed. If the proxy shows edge, futures
  order-flow data is the validation step before any live use.
- Period: 2021-04 -> 2026-08 (aligns with funnel OOS boundary 2025-11-01).
- Bar construction: footprint bars from ticks (bid/ask volume per price bucket, delta =
  askvol-bidvol... aggressor = trade direction from tick sequence), plus OHLCV.

## Pre-registered bars (same conventions as every cycle)
- OOS: ts >= 2025-11-01 fixed.
- n_OOS >= 150 trades across tested symbols.
- GO iff bootstrap(10k seed 42) P(meanR_OOS > 0) >= 0.95 AND meanR_OOS >= 0.30R AND
  PF_OOS >= 1.5 AND OOS avgR beats window-funnel reference (+0.455R/PF 1.93 same slice).
- KILL at n>=150 on any failed bar; INCONCLUSIVE below.
- Failure path (pre-committed): if the free data feed is infeasible (reachability, size,
  licensing ambiguity) OR the footprint edge cannot be implemented point-in-time within the
  data budget, the experiment is recorded KILLED-ON-DATA with the reason — no silent swaps.

## Deliverable order
1. Data pipeline proof: 1 week US100.IDX ticks downloaded, parsed, footprint + delta valid.
2. Full 5y backfill (background, ~1-3GB).
3. Selector + sim (his trigger, both directions), same conservative exits as the bot.
4. OOS stats -> verdict. Nothing live, ever, until GO + separate futures-data validation.
