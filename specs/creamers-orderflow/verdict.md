# CREAMER ORDERFLOW v2 — verdict (2026-09-04) — KILLED-ON-DATA

Spec: SPEC.md (pre-registered before data build) + ENGINE_PARAMS.md (parameters and
roster locked pre-run). Run identity: hermes session bg_125245_fed2c0 (user request:
"complete this strategy and check how it performs in live market").

## What was completed
1. Feed hunt (exhaustive, free/no-signup only):
   - datafeed.dukascopy.com bi5: EURUSD works (forex); US30.IDX/US100.IDX/US500.IDX/
     US2000.IDX -> 404/URLError on all hosts (www/datafeed/freeserv). Index CFDs are
     NOT on the public bi5 feed.
   - dukascopy-python platform codes (E_NQ-100, E_SandP-500...) are trading-platform
     names, not datafeed paths.
   - dukascopy-node JETTA API (jetta.dukascopy.com/v1, no key) — RESOLVED the symbol
     question: codes USATECH.IDX-USD (Nasdaq-100, ticks since 2013), USA500.IDX-USD
     (S&P-500, since 2012), USA30.IDX-USD, USSC2000.IDX-USD, plus real ETFs
     (SPY.US-USD, QQQ.US-USD, DIA.US-USD, IWM.US-USD). JSON per UTC hour; parse
     verified (2026-08-20 13h USATECH: 27,680 ticks, prices ~29,25x tracked the
     index, format = ms inter-arrival deltas + int price deltas x 0.001 multiplier).
2. Pipeline proof (SPEC deliverable 1): fetch -> parse -> 3-min/5-min footprint bars
   with per-level buy/sell and net delta. Engine harness ready (footprint.py in
   ~/creamer-live/).
3. Volume/delta VALIDATION (the step that killed it):
   - USATECH.IDX-USD (the SPEC's NQ proxy): askVolumes CONSTANT 290.0 and bidVolumes
     CONSTANT 220.0 for all 27,680 ticks in the hour -> delta is a fixed +70/tick,
     ZERO information. Synthetic CFD quote sizes, not trades.
   - USA500.IDX-USD: same constant pattern (SPY.US-USD too: constant 12,000/tick,
     delta exactly 0).
   - QQQ/DIA/IWM: coarse 3-5 distinct size levels/hour (900/90,000/120,000 lots) —
     aggregated display sizes, not print-level tape. Bar-delta vs bar-return on a
     full RTH day (130 x 3-min bars): QQQ r=+0.241 (sign-agree 0.60), DIA r=+0.019,
     IWM r=+0.157. Weak-to-noise directional content; NOT real aggressor delta.

## Pre-registered failure path — TRIGGERED
SPEC.md clause: "if the free data feed is infeasible ... OR the footprint edge cannot
be implemented point-in-time within the data budget, the experiment is recorded
KILLED-ON-DATA with the reason — no silent swaps."
Reason: the free feed class that resolves WITHOUT signup (Dukascopy JETTA) carries
REAL bid/ask PRICES but SYNTHETIC/CONSTANT/COARSE VOLUMES for every equity-index
instrument. The v2 premise ("aggressor-side delta is real bid/ask tick attribution")
is FALSE as measured. Creamer's trigger IS the delta/absorption layer (concentration
at the extreme + one-sided delta + imbalance). Running the OOS test on this delta
would produce a verdict that is not a test of his strategy — the same category error
the user caught on v1. Verdict: KILLED-ON-DATA, not KILL (no strategy verdict exists
on non-real delta; the door stays open for real data).

## What would unblock (real paths, recorded for the user's call)
1. Databento free tier (CME MDP3 real NQ/ES/RTY/YM futures ticks, real trades/bid-ask,
   real delta) — ONE account signup + API key (user's standing no-signup rule; his
   call). Free credits buy a limited historical slice (~months of NQ trades/bbo),
   enough for a first honest read, likely INCONCLUSIVE at the 150-trade bar; full
   OOS window would exceed free credits (paid, small $).
2. Paid order-flow/feed ~$100-300/mo (footprint futures data, e.g. Sierra/Bookmap
   historical, or Databento subscription) — full 5y window, real verdict possible.
Either path goes through a fresh pre-registered spec (constitution: no silent data
swaps). Nothing was wired live; nothing in live config was touched.

## Evidence files
- ~/creamer-live/footprint.py (working jetta parser + footprint builder)
- ENGINE_PARAMS.md (locked parameters + resolved feed + roster)
- Probe logs in this session (constant volumes, correlation tests)

Recorded 2026-09-04 by Hermes session bg_125245_fed2c0.
