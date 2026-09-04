# Validation Memo: close the direction question — the signal exists, the profit doesn't

Date: 2026-09-01 | Source: specs/conditioned-direction/direction-report.md

## ONE recommendation: NO CHANGE — do not add a direction component.
The direction question is closed with numbers: post-signal momentum is real
(r≈0.30, P=1.0 on the hold-out) but does NOT improve the funnel (aligned
book -0.014 vs baseline +0.027 avgR; August worse). The funnel already
captures the signal; costs and exits eat the residual.

## Why this is the final answer
1. This is the strongest direction result the project has ever measured —
   65% hit-rate, out-of-sample, verified at three levels — and it STILL
   fails the book test. If the best direction signal in 5.3 years of data
   cannot improve the funnel, no direction component will.
2. The funnel's own selection already encodes the momentum (EMA cross +
   ADX gate fire in momentum-conducive states — that's why the signals have
   the 65% edge in the first place). Adding a direction overlay would be
   paying twice for the same information, minus costs.
3. The doctrine's ceiling (r≈0.27-0.39) is now measured end-to-end: the
   pulse exists at the top of that envelope, and it is not additive.

## What I will NOT do
Add a direction overlay. Retrain models to predict direction (the forbidden
path). Re-open this question without a fundamentally new data type (e.g.,
order-flow or cross-asset data — the ONLY thing that could change this
answer, per the doctrine's own framing).

## Watch item (not an action)
The 0.50-0.65 proba zone (new under the user override) had no forward
samples yet; as live trades accumulate, the audit can be re-run — but per
this verdict, even a positive zone result would face the same book test.

## The one thing that could legitimately revisit this
A new INFORMATION SOURCE, not a new model: order-flow, volume profile, or
cross-market data (e.g., bond yields vs NQ). The doctrine says price-only
predictability is the ceiling; new data types are the only unmeasured
dimension. If the user ever wants that, it is a new pre-registered cycle —
with this report as the baseline to beat.
