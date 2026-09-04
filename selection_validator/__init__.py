"""selection_validator — SELECTION-VALIDATION experiment (spec-kit 2026-08-30).

Proves or kills the veto-quality band selector (baseline + quality >= q)
against the current funnel baseline, on the pre-registered OOS slice
(entry ts >= 2026-08-01), one-sided bootstrap p<0.05 (10k draws, seed 42).

Package layout mirrors specs/selection-validation/{spec,plan,tasks}.md.
"""
