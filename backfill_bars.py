#!/usr/bin/env python3
"""Clean backfill: TopstepX history API unit semantics = SECONDS (unit=1, unitNumber=180 = 3min).
Merge into restored (clean) data/{SYM}_3min.csv with atomic replace. Verify alignment after.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from broker import make_broker

SYMS = ["NQ", "ES", "RTY", "YM", "GC"]
DATA = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data")

b = make_broker()
b.authenticate()

for sym in SYMS:
    try:
        cid = b.get_active_contract(sym, live=False)["id"]
        path = f"{DATA}/{sym}_3min.csv"
        old = pd.read_csv(path)
        old["datetime"] = pd.to_datetime(old["datetime"], utc=True)
        last = old["datetime"].iloc[-1]
        end = pd.Timestamp.now(tz="UTC")
        cur = last - pd.Timedelta(hours=6)  # overlap for safety
        frames = []
        while cur < end:
            nxt = min(cur + pd.Timedelta(days=3), end)
            r = b._post("/History/retrieveBars", {
                "contractId": cid, "live": False,
                "startTime": cur.isoformat(), "endTime": nxt.isoformat(),
                "unit": 1, "unitNumber": 180,  # SECONDS semantics -> 3-min bars
                "limit": 5000, "includePartialBar": False,
            })
            df = pd.DataFrame(r.get("bars", []))
            if not df.empty:
                frames.append(df)
            cur = nxt
        if not frames:
            print(sym, f"NO NEW BARS (last={last})", flush=True)
            continue
        new = pd.concat(frames).rename(columns={
            "t": "datetime", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        new["datetime"] = pd.to_datetime(new["datetime"], utc=True)
        new = new.sort_values("datetime").drop_duplicates("datetime")
        comb = pd.concat([old, new]).sort_values("datetime").drop_duplicates("datetime")
        fd, tmp = tempfile.mkstemp(dir=DATA, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                comb.to_csv(f, index=False)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        # verify: no sub-minute diffs in new region
        comb2 = comb.copy()
        d = comb2["datetime"].diff().dt.total_seconds().div(60)
        submin = int((d < 1.0).sum())
        print(sym, f"old={len(old)} new={len(new)} total={len(comb)} submin={submin} last={comb['datetime'].iloc[-1]}", flush=True)
    except Exception as e:
        print(sym, "ERROR", repr(e), flush=True)
