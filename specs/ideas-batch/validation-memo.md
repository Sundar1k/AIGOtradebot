# Validation Memo: keep the full window; first-hour narrowing is the next re-test

Date: 2026-09-01 | Source: specs/ideas-batch/report.md

## ONE recommendation: NO CHANGE now — keep the full 09:30-12:00 window.
The first-hour (09:30-10:30) narrowing is the single idea with a pulse
(+0.404 avgR hold-out, +0.51 in August — the only profitable thing in
August anywhere in this project) but n=89 < 150 and P=0.554: INCONCLUSIVE
per the pre-registered bar. Gap, jump, and volume filters are dead —
archived with numbers.

## Why wait (the rule, not the vibes)
The August +0.51 number is the most seductive single result of the project.
It is ALSO one month with 89 trades. The pre-registered bar — written before
any of this ran — says no GO below 150. Wiring the first hour now would be
exactly the "close enough" move every cycle exists to prevent.

## The plan (pre-registered continuation)
1. Keep the live config unchanged (full window, 0.40/0.65/2.0).
2. The first-hour subset of live trades accumulates into the dataset; when
   the hold-out slice holds >=150 first-hour trades, re-run
   `python -m selection_validator.ideas_batch` (same bars, same rule).
3. If the four bars then pass -> propose narrowing the live window to
   09:30-10:30 (memo + user approval). If not -> KILL, archived.

## Watch item
The August resilience (+0.51) could be a real property (the opening hour's
trend-initiation edge) or a single-month artifact. Only the growing sample
can tell — and the machinery is ready to ask it.
