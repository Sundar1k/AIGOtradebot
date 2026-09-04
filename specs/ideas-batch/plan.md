# Implementation Plan: Ideas Batch

**Branch**: `07-ideas-batch` | **Date**: 2026-09-01 | **Spec**: `specs/ideas-batch/spec.md`

## Summary

Four cheap selection ideas (gap filter x2 variants, jump filter, volume
confirmation, first-hour narrowing) tested on the existing dataset + bar
CSVs with a common pre-registered hold-out bar. No replay, no GPU, ~20 min.

## Technical Context

Python 3.12 (bot venv); pandas/numpy (existing); pytest (existing).
Reads selection_validator/data/signals_*.jsonl + data/*_3min.csv; writes
specs/ideas-batch/ + selection_validator/results/ideas_batch.json.

## Constitution Check

All 8 principles PASS: pre-registered (II), frozen loop (III), hold-out
bootstrap (IV), memo-gated (VI), dataset-only (VII), transparent (VIII).

## Phases

1. Feature caches per signal (US1): gap (open[day] - close[prev day]),
   jump (max |cc| over last 2 bars vs 1.5x ATR20), volume ratio
   (vol[bar]/mean(vol[20])), hour sub-window. Point-in-time, cached per
   (symbol, ts), spot-checked.
2. Candidate books (US2): each variant applied ON TOP of the current funnel
   (window + floor 0.40 + ceil 0.65); train-fold tables (report) + hold-out
   decision.
3. Report + memo (US3): ideas_batch.json + report.md + validation-memo.md;
   commit; git-diff check; tasks ticked.

## Reuse

selection_validator.dataset/harness/time_window; the bar-at-ts cache pattern
from mechanics_audit; cycle-4 folds.

## Complexity

None — four filter masks + the standard bootstrap bar.
