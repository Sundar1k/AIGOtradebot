#!/usr/bin/env bash
# run_selection_validation.sh — end-to-end Selection Validator (spec plan).
# Stages: replay (all symbols) -> quality scoring (taken signals) -> verdict.
# Each stage is idempotent; re-running skips nothing but re-does the stage.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=selection_validator/data/run.log
mkdir -p selection_validator/data
echo "=== selection validation $(date -Is) ===" | tee -a "$LOG"

echo "[1/3] replay (5 symbols, ~6h) —" | tee -a "$LOG"
HF_HUB_OFFLINE=1 .venv/bin/python -m selection_validator.replay >> "$LOG" 2>&1

echo "[2/3] quality scoring (taken signals via veto :8765) —" | tee -a "$LOG"
HF_HUB_OFFLINE=1 .venv/bin/python -m selection_validator.quality_score >> "$LOG" 2>&1

echo "[3/3] verdict —" | tee -a "$LOG"
HF_HUB_OFFLINE=1 .venv/bin/python -m selection_validator.verdict >> "$LOG" 2>&1

echo "=== done: specs/selection-validation/verdict.md ===" | tee -a "$LOG"
