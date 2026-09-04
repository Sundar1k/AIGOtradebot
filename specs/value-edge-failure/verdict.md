# VALUE-EDGE FAILURE — verdict (2026-09-04)

Spec: SPEC.md (pre-registered before code). Engine: value_edge_failure.py.
Code-fix rerun (allowed by spec): short-side simulator risk was long-only (E-stop), silently
dropping every short flip; fixed to direction-aware (stop-E) before the full run.

## Data
5 symbols x 3-min RTH bars 2021-04 -> 2026-08 (clean re-backfill). 1,310 signals
(819 unique trade events; each trade simulated vs 2 pre-registered target variants).
OOS slice: ts >= 2025-11-01 (fixed, funnel boundary).

## Full-period
| variant | n | meanR | medianR | WR | sumR | PF |
|---|---|---|---|---|---|---|
| A_poc (target = prior POC) | 464 | +0.025 | +0.142 | 69.0% | +11.5 | 1.08 |
| B_swing (target = prior swing) | 846 | -0.033 | -0.664 | 41.3% | -27.7 | 0.94 |

## Out-of-sample (ts >= 2025-11-01)
| variant | n | meanR | WR | sumR | PF | P(meanR>0) bootstrap 10k seed42 |
|---|---|---|---|---|---|---|
| A_poc | 59 | +0.085 | 79.7% | +5.0 | 1.45 | 0.839 |
| B_swing | 111 | -0.160 | 38.7% | -17.7 | 0.67 | 0.042 |

OOS unique trade events: 109.

## Pre-registered GO bars
- n_OOS >= 150: FAIL (109 unique; 59 A_poc / 111 B_swing)
- P(meanR>0) >= 0.95: FAIL (0.839 / 0.042)
- meanR_OOS >= +0.30R: FAIL (+0.085 / -0.160)
- PF_OOS >= 1.5: FAIL (1.45 / 0.67)
- Beat window-funnel reference on same slice (+0.455R / PF 1.93): FAIL (both)
- KILL/Failure-path clause (OOS sits at/below funnel baseline): TRIGGERED both variants.

## Verdict: KILL (both variants)
The price/volume-location proxy carries NO selection edge over the existing funnel on
3-min OHLCV bars. B_swing is negative OOS with P(positive)=0.042 (96% likely negative).
A_poc shows the classic small-target/wide-stop trap (WR ~70-80% but meanR ~0 because the
target at POC sits near entry while risk is defined by the failed extreme) — the same shape
the regime-halt and vol audits flagged before. Without footprint/delta (bid/ask data we do
not own), the Creamer structure loses its trigger; what remains is a mean-reversion-to-value
already priced inside the funnel's costs.

Recorded 2026-09-04. Direction question stays closed. New data types (real footprint/delta)
would be the only revisit path, and they are not in the current data budget.
