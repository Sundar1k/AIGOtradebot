# Implementation Plan: Selection Validator

**Branch**: `01-selection-validator` | **Date**: 2026-08-30 | **Spec**: `specs/selection-validation/spec.md`

**Input**: Feature specification from `specs/selection-validation/spec.md`

## Summary

Build a selection validator that proves (or kills) the quality-band selection
lever on the bot's own data. It re-scores every graded signal the engine has
ever produced — backtest replay through the existing handle_bar, plus live and
paper ledgers — evaluates the current funnel (baseline: floor 0.35 / ceil 0.50
+ gate stack) against a candidate (baseline + veto-quality ≥ q, q ∈ {5,6,7}),
and decides on a pre-registered OOS slice (entry ts ≥ 2026-08-01) with a
one-sided bootstrap (10k draws, seed 42, p<0.05). GO requires N≥150 OOS trades,
P(ΔavgR>0)>0.95 and PF_candidate ≥ PF_baseline; anything else is KILL or
INCONCLUSIVE. The live loop is frozen during the experiment (doctrine).

## Technical Context

**Language/Version**: Python 3.12 (repo .venv)
**Primary Dependencies**: pandas, numpy, xgboost (existing); pytest 9.1.1 (existing)
**Storage**: JSONL ledger (`selection_validator/data/scored_signals.jsonl`) +
results dir; existing CSVs (data/*_3min.csv, backtest CSVs) read-only
**Testing**: pytest, tests/ (repo has conftest.py; follow existing style)
**Target Platform**: Linux (this box)
**Project Type**: research harness (CLI), library-shaped for reuse
**Performance Goals**: replay of 5y×5 symbols through handle_bar in < 30 min;
bootstrap 10k draws per slice in < 10 s
**Constraints**: point-in-time only (no future bars in features); fixed seed 42;
no modification of bot.py / supervisor.py entry logic; offline (HF_HUB_OFFLINE=1)
**Scale/Scope**: ~1k-100k scored signals, 2 selectors, 1 verdict

## Constitution Check

*GATE: must pass — verified against .specify/memory/constitution.md.*

| Principle | Status |
|-----------|--------|
| I. Evidence over prediction | PASS — no direction model, selection only |
| II. Pre-registered experiment | PASS — success bar + kill rule in spec before code |
| III. Settled doctrine | PASS — 3-min, band, loop all frozen (FR-005) |
| IV. Validation gates | PASS — N≥150, bootstrap p<0.05, OOS slice fixed at spec time |
| V. Approval before build | PASS — this plan + spec + tasks go to the user gate first |
| VI. Fail-closed ops | PASS — feed-only wiring (US5), never auto-gates |
| VII. Evidence-budget match | PASS — reuses existing infra; one new module |
| VIII. Honest accounting | PASS — slippage/commission model reused as-is, labeled |

## Project Structure

### Documentation (this feature)

```text
specs/selection-validation/
├── spec.md              # This feature's spec (approved gate)
├── plan.md              # This file
├── tasks.md             # /speckit-tasks output (next gate)
└── verdict.md           # Written at the end — numbers-backed decision
```

### Source Code (repository root)

```text
selection_validator/          # NEW — research harness (library-shaped)
├── __init__.py
├── dataset.py               # ScoredSignal schema, replay+ledger ingestion, leak-check
├── selectors.py             # BaselineSelector + CandidateSelector (shared evaluator)
├── harness.py               # bootstrap comparison, verdict computation
└── verdict.py               # verdict.md writer (numbers, not prose)

tests/
├── test_dataset.py          # leak-check, schema, null-outcome handling
├── test_selectors.py        # baseline sanity, q-threshold behavior
└── test_verdict.py          # synthetic GO / KILL / INCONCLUSIVE

scripts/
└── run_selection_validation.sh   # end-to-end runner (replay → ingest → eval → verdict)
```

**Structure Decision**: single-project layout matching the existing flat repo
style (modules at root, tests/ at root — no src/ indirection). The validator
is one importable package `selection_validator/`; scripts/ holds the runner
(mirrors finetune/ precedent).

## Phase Breakdown

- **Phase 0 — Dataset**: backtest replay with signal logging (extend
  backtest.py with `--log-signals`), live/paper ledger ingestion, leak-check.
- **Phase 1 — Baseline sanity**: BaselineSelector reproduces +0.58R / PF 2.11
  ± tolerance on pre-2026-08-01 history; any miss = harness bug, fix before
  proceeding (no verdicts on a broken harness).
- **Phase 2 — Candidate**: quality-band sweep q ∈ {5,6,7} on the SAME
  historical slice (measurement only — decision power comes from OOS).
- **Phase 3 — Verdict**: OOS slice (≥ 2026-08-01) bootstrap at seed 42;
  verdict.md with per-slice numbers; GO / KILL / INCONCLUSIVE.
- **Phase 4 — Feed wiring**: AUTOTRADE_SELECTOR=feed in paper_live.py
  (log-only), forward OOS accumulation, promotion proposal only.
- **Phase 5 — Polish**: docs, commit per task, verdict archived.

## Reuse (no reinvention)

- `edge_monitor.bootstrap_p_lt` — bootstrap (10k, seeded) already implemented.
- `evolve.winrate_by_quality` — quality-band prior evidence.
- `backtest.py` — replay driver (add `--log-signals`); trade CSV writer exists.
- `paper_live.py` — forward feed wiring point.
- `finetune/` — precedent for experiment docs + result files.

## Complexity Tracking

None — no constitution violations. The new surface is one package + tests;
everything else is reuse. Complexity budget is respected (Principle VII).
