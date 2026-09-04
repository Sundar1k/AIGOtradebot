# Direction Report: Regime-Conditioned Direction (cycle 6)

Generated: 2026-09-01 | Dataset: 98,363 rows, 11,364 funnel signals | Forward
K-bar directions from bar CSVs (point-in-time) | Horizons K=10 (30m), K=30
(90m) | 12 pre-registered conditions x 2 horizons = 24 candidates | Folds:
cycle-4 walk-forward split (pre-2025-11-01) | Hold-out: ts >= 2025-11-01
(cycle-3 boundary) | Results: selection_validator/results/direction_audit.json

## The answer to the direction question: there IS a real signal — and it is
NOT tradeable on top of the funnel. Verdict: **KILL** (per the pre-registered
rule: a hit-rate edge only counts if it improves the book).

## Finding 1 — the pulse is real (verified, not an artifact)

Post-signal 30-minute momentum continuation runs at ~65% (r≈0.30):
- ALL bars (raw market): 49.3% — a coin flip (the doctrine's old finding)
- Funnel SIGNAL bars (EMA cross/ORB, trend-gated): 65.0%
- Per symbol: YM 68.3%, NQ 67.4%, ES 66.5%, RTY 62.7%, GC 59.8%
- Hold-out (Nov 2025-Aug 2026): P(hit-rate > 0.50) = 1.0000, n = 629

Verified at three levels (raw bars vs signal bars vs per-symbol) — not a code
artifact. The implied correlation (2 x 0.65 - 1 = 0.30) sits exactly at the
doctrine's measured ceiling (best r≈0.27 live, vol r≈+0.39). The signals fire
in momentum-conducive states (strong EMA cross + ADX >= 18), and in those
states the next 30 minutes continue more often than not. This is the ONLY
direction result in the project's history that beats a coin flip out-of-sample.

## Finding 2 — the pulse does NOT improve the book (the kill)

The pre-registered rule requires the conditioned book to beat the funnel:

| hold-out (Nov 2025-Aug 2026) | trades | avgR | PF |
|---|---|---|---|
| baseline funnel | 1,567 | +0.027 | 1.03 |
| momentum-aligned only | 627 | -0.014 | 0.98 |
| August aligned vs August baseline | -1.280 | vs -1.218 | worse |

The momentum-aligned subset was NOT better than the funnel — slightly worse.
Why: the funnel's own direction selection (EMA/ORB direction + model grading)
already encodes this momentum; aligning by momentum removed winners as often
as losers. The edge exists but is already captured; the residual is eaten by
slippage and the exits — exactly the doctrine's recorded lesson ("costs eat
it").

## All 24 candidates (transparent)

Momentum-family conditions (continuation prediction) dominated: flow_vol_k10
(median 0.673), in_window_k10 (0.659), momentum_k10 (0.651), adx_hi_k10
(0.651), proba_lo_k10 (0.651), out_window_k10 (0.650), chop_vol_k10 (0.642),
then the K=30 family (0.56-0.63). All passed fold selection (>=0.53 in 90%
of folds). The model's r_hat sign carried a weak signal (0.51-0.53, failed
fold selection). EMA alignment and RSI mean-reversion were coin-flips or
worse (0.43-0.45). The 0.50-0.65 proba zone (new under the user override)
had insufficient forward samples (NaN) — will accumulate live.

## The direction question is now CLOSED

1. Unconditional direction: dead (46-53%, every model family — confirmed).
2. Conditional direction (post-signal momentum, trend-gated): REAL at
   r≈0.30, out-of-sample — the strongest direction result ever measured in
   this project.
3. Tradeable direction: NO — it does not add to the funnel after costs.
   The book test is the only verdict that matters, and it says KILL.

The user's original dream — "make the bot predict chart direction" — now has
a definitive, measured tombstone: the signal exists, the profit doesn't.
The funnel remains the edge; direction remains a coin flip AFTER costs.
