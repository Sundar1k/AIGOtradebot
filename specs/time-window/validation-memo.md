# Validation Memo: wire the 09:30-12:00 ET entry window into the live bot

Date: 2026-08-31 | Source: specs/time-window/verdict.md (pre-registered replay)

## ONE recommendation: WIRE IT — gate NEW ENTRIES to 09:30-12:00 ET
(proposal only; requires user approval before any live change)

## Why GO is trustworthy
- OOS (Nov 2025-Aug 2026, never used to choose the window): +0.394R / PF 1.75
  vs +0.027R / PF 1.03 all-day. P(ΔavgR>0) = 0.9997. Four of four
  pre-registered bars passed.
- The effect is present in every year 2021-2026 (stability tables) and the
  gate has zero tunable parameters.
- August damage would have been cut ~77% (36 trades instead of 482; -31R vs
  -587R) — the regime that sank the account is exactly what this window
  sidesteps.

## The proposed live change (entries only)
- In supervisor.py: block NEW entries when ET time-of-day is outside
  [09:30, 12:00). Position MANAGEMENT (trail, stop, exits) continues 24h —
  open trades are never orphaned.
- Time handling: America/New_York wall time (zoneinfo), NOT a fixed UTC
  offset — DST-correct year-round (the replay used UTC-4 for the summer
  dataset; the live gate must be TZ-aware).
- Symbols: all five (the window effect is per-symbol positive in stability
  tables; no per-symbol carve-outs in this cycle).
- Env-gated (AUTOTRADE_ENTRY_WINDOW=1 default OFF for now?) — recommend
  starting with the gate ON for the paper book first, then live after a
  paper confirmation window. This is a flow change; paper-first matches
  the doctrine's validation culture.

## Honest caveats
- Volume: the window keeps ~21% of funnel trades (344/1,567 on OOS) — the
  N>=150 validation problem gets harder, not easier. The bot will trade
  rarely (~1-2 entries/day expected). Its edge per trade is better; its
  sample growth is slower.
- This validates the ENTRY gate on replay; live fills/regimes may differ —
  hence paper-first.
- The first hour (09:30-10:30) carries most of the edge (OOS +0.64 vs +0.27
  second hour). If the full window ever needs tightening, that is a NEW
  pre-registered experiment, not a knob.

## What I will NOT do without you
Change any live config today. This memo is the proposal; the wiring (paper
first, then live) happens only on your explicit go.
