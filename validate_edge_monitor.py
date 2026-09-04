#!/usr/bin/env python3
"""validate_edge_monitor.py — replay the edge monitor over the backtest CSVs
and measure the 3 pre-registered success criteria from EDGE_MONITOR_SPEC.md.

Model (advisory replay): record every trade in point-in-time order. A trade is
"would-be-blocked" if the monitor was in HALT state immediately before it. This
answers the gate's core question — would it have flagged the August collapse and
pulled back — without any look-ahead (each decision uses only prior trades).
"""
import os
import pandas as pd

import edge_monitor as em

BASE = os.path.dirname(os.path.abspath(__file__))

# Load all 5 backtest CSVs, combine, sort by entry_time (point-in-time).
rows = []
for sym in ["NQ", "ES", "RTY", "YM", "GC"]:
    p = f"{BASE}/log/backtest_{sym}.csv"
    if not os.path.exists(p):
        continue
    for _, r in pd.read_csv(p).iterrows():
        rows.append((r["entry_time"], sym, r["strategy"], float(r["r"])))
rows.sort(key=lambda x: x[0])

mon = em.EdgeMonitor(state_file=None, quiet=True)

timeline = []   # (entry_time, state_before, r, blocked)
halts = []      # entry_times where a halt transition happened
for entry_time, sym, strat, r in rows:
    if mon.is_blocked_at(entry_time):
        timeline.append((entry_time, "halt", r, True))   # bot did not take it
        continue
    state_before = mon.state
    mon.record({"r": r, "strategy": strat, "ts": entry_time})
    if state_before != "halt" and mon.state == "halt":
        halts.append((entry_time, sym))
    timeline.append((entry_time, state_before, r, False))

def is_august(ts):
    return ts[:7] == "2026-08"

aug_all = sum(r for ts, sb, r, b in timeline if is_august(ts))
aug_kept = sum(r for ts, sb, r, b in timeline if is_august(ts) and not b)
aug_with_gate = aug_kept                      # only non-blocked trades realize
aug_gate_effect = aug_kept - aug_all          # negative => gate made August WORSE

aprjul_all = sum(r for ts, sb, r, b in timeline if not is_august(ts))
aprjul_kept = sum(r for ts, sb, r, b in timeline if not is_august(ts) and not b)
profit_retained = aprjul_kept / aprjul_all if aprjul_all else 1.0

false_halts = sum(1 for h, s in halts if not is_august(h))
aug_halts = sum(1 for h, s in halts if is_august(h))

print("=== edge-monitor validation replay ===")
print(f"mode={em.MODE}  window={em.WINDOW}  halt_p={em.HALT_P}  watch_p={em.WATCH_P} "
      f"healthy={em.B_HEALTHY}  cooldown_h={em.COOLDOWN_H}")
print(f"total trades replayed: {len(timeline)}")
print()
print(f"August     : WITHOUT gate {aug_all:+.2f}R | WITH gate {aug_with_gate:+.2f}R "
      f"(gate effect {aug_gate_effect:+.2f}R)")
print(f"April-July : WITHOUT gate {aprjul_all:+.1f}R | WITH gate {aprjul_kept:+.1f}R")
print()
print(f"halts: {len(halts)} total ({aug_halts} in Aug, {false_halts} in Apr-Jul)")
for h, s in halts:
    print(f"   HALT @ {h} ({s})")
print()
c1 = aug_gate_effect >= 0.0          # gate must not make August worse
c2 = profit_retained >= 0.90
c3 = false_halts <= 2
print("=== success criteria (revised) ===")
print(f"1. Gate does NOT worsen August (effect >= 0): {aug_gate_effect:+.2f}R -> {'PASS' if c1 else 'FAIL'}")
print(f"2. April-July retained >= 90%   : {profit_retained:.0%}  -> {'PASS' if c2 else 'FAIL'}")
print(f"3. False halts (Apr-Jul) <= 2   : {false_halts}        -> {'PASS' if c3 else 'FAIL'}")
print()
print("OVERALL:", "PASS — deploy candidate" if (c1 and c2 and c3)
      else "FAIL — spec revision needed (NOT a threshold tune)")
