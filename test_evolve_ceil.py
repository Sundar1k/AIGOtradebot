#!/usr/bin/env python3
"""Test the Evolver's confidence-ceiling learning: tighten on high-proba
losses, ease back on high-proba wins, respect bounds and cooldowns."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evolve


def fresh(name):
    p = f"/tmp/{name}.json"
    if os.path.exists(p):
        os.remove(p)
    return p


def trade(r, proba):
    return {"r": r, "strategy": "ema", "side": "LONG", "entry": 100,
            "exit": 100 + r * 5, "quality": 6, "proba": proba,
            "ts": "2026-08-17T00:00:00+00:00"}


# 1) high-proba losses -> tighten
e = evolve.Evolver(0.30, state_file=fresh("t1"))
for _ in range(10):
    e.record(trade(-1.0, 0.72))
st = e.status()
print("t1 after 10 high-proba losses: ceil =", st["ceil"], "band_wr =", st["high_band_winrate"])
assert st["ceil"] < 0.50, "should tighten below base 0.50"

# 2) high-proba wins (fresh window) -> ease back up
e.trades.clear()
e.last_ceil_change = 0.0
for _ in range(10):
    e.record(trade(2.0, 0.72))
st = e.status()
print("t2 after 10 high-proba wins:    ceil =", st["ceil"], "band_wr =", st["high_band_winrate"])
assert st["ceil"] > 0.40, "should ease above tightened floor"

# 3) bounds: never below CEIL_FLOOR
e3 = evolve.Evolver(0.30, state_file=fresh("t3"))
e3.trades.clear()
e3.last_ceil_change = 0.0
e3.ceil = 0.50
for _ in range(5):
    e3.record(trade(-1.0, 0.72))
print("t3 ceil after repeated losses:", e3.status()["ceil"])
assert e3.status()["ceil"] >= evolve.CEIL_FLOOR, "never below CEIL_FLOOR"

# 4) persistence round-trip
st_before = e.status()
e2 = evolve.Evolver(0.30, state_file=e.state_file)
assert e2.status()["ceil"] == st_before["ceil"], "ceil must persist"
print("t4 persistence OK:", e2.status()["ceil"])

print("\nALL EVOLVER CEILING TESTS PASSED")
