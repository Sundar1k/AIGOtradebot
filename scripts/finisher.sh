#!/usr/bin/env bash
# finisher.sh — waits for all replays, then quality-scores and runs the verdict.
# Launched detached; survives this session.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=selection_validator/data/finisher.log
echo "=== finisher $(date -Is) — waiting for replays ===" >> "$LOG"

while pgrep -f "selection_validator.repl[a]y" > /dev/null 2>&1; do
  sleep 60
done
echo "=== replays done $(date -Is) — dedup ===" >> "$LOG"
HF_HUB_OFFLINE=1 .venv/bin/python -m selection_validator.dedup_signals >> "$LOG" 2>&1
echo "=== dedup done $(date -Is) — quality pass ===" >> "$LOG"
HF_HUB_OFFLINE=1 .venv/bin/python -m selection_validator.quality_score >> "$LOG" 2>&1
echo "=== quality done $(date -Is) — verdict ===" >> "$LOG"
HF_HUB_OFFLINE=1 .venv/bin/python -m selection_validator.verdict >> "$LOG" 2>&1
echo "=== verdict done $(date -Is) ===" >> "$LOG"
