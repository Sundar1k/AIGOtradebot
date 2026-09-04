#!/usr/bin/env python3
"""reflection.py — decision memory (feature #5).

Collects every closed trade's veto reason + outcome into a human-readable
lessons file (~/.autotrade_lessons.md), TradingAgents-style
reflection: numbers tell you WHAT happened, the veto reason tells you WHY
the bot thought it. The attribution agent reads this file alongside the
numeric ledger so weekly rule changes consider the narrative too.

Format (one entry per closed trade):
    ## 2026-08-17 18:00 UTC — ES SHORT 1.0R
    - entry 7817.25 exit 7815.84 | target
    - veto reason: "EMA downtrend; stochastic pullback down: 2-of-3 agreement"
    - outcome: WIN | lesson hint: (auto)

Append-only; never rewritten. Bounded to the last N entries.
"""
import datetime as dt
import json
import os

LESSONS = os.path.join(os.path.expanduser("~"), ".autotrade_lessons.md")
MAX_ENTRIES = 400


def _read() -> str:
    try:
        with open(LESSONS) as f:
            return f.read()
    except Exception:
        return ""


def record(trade: dict):
    """trade: {r, strategy, side, entry, exit, quality, veto_reason, ts,
    exit_kind} — called on every trade close (wired as on_trade_close)."""
    try:
        r = float(trade.get("r", 0.0))
        side = trade.get("side", "?")
        sym = trade.get("symbol", "?")
        reason = (trade.get("veto_reason") or "").strip() or "(no veto reason)"
        ts = (trade.get("ts") or dt.datetime.now(dt.timezone.utc).isoformat())[:16]
        outcome = "WIN" if r > 0 else "LOSS" if r < 0 else "FLAT"
        entry = trade.get("entry", 0)
        exit_ = trade.get("exit", 0)
        kind = trade.get("exit_kind", "?")
        line = (f"\n## {ts} UTC — {sym} {side} {r:+.2f}R ({outcome})\n"
                f"- entry {entry} exit {exit_} | {kind}\n"
                f"- veto reason: \"{reason}\"\n")
        body = _read() + line
        # keep only the last MAX_ENTRIES '## ' blocks
        blocks = body.split("\n## ")
        if len(blocks) > MAX_ENTRIES + 1:
            body = "\n## " + "\n## ".join(blocks[-(MAX_ENTRIES + 1):])
        with open(LESSONS, "w") as f:
            f.write(body)
    except Exception as e:
        print(f"reflection write failed: {e}", flush=True)


def latest(n: int = 20) -> str:
    """Last n lesson entries as text (for the attribution agent / reporting)."""
    blocks = _read().split("\n## ")
    if not blocks or blocks == [""]:
        return "(no lessons yet)"
    return "\n## ".join(blocks[-(n + 1):])
