import os
#!/usr/bin/env python3
"""orb_backtest.py — test the DOCUMENTED ORB edge on our own 5y data.

Variants (from PROVEN_STRATEGIES_RESEARCH.md, pre-registered before running):
  V1 baseline: break of 30m opening range, either direction
  V2 long-only: same but LONG side only
  V3 close-confirm: V2 + require a 5m bar CLOSE beyond the range high
  V4 time-window: V3 + entry only 09:30-11:00 ET (first 90 min)
  V5 full documented: V4 + target = 1x range width, stop = other side of range

Bracket: stop = opposite side of range, target = 1x range width (~1:1 RR as
documented). Also report a 2R variant for our book's geometry.

Data: data/{SYM}_5min.csv (fresh download) + {SYM}_15min.csv for context.
Split: chronological — stats on FULL period, then last-year-only stability.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
TICK = {"NQ": 0.25, "ES": 0.25, "RTY": 0.10, "YM": 1.0, "GC": 0.10}


def load_5m(sym):
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_5min.csv"),
                     parse_dates=["datetime"]).rename(columns={"datetime": "time"})
    df = df.sort_values("time").reset_index(drop=True)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    # ET conversion via -4h (EDT) approximation; session filter uses UTC hour
    return df


def rth_session(df):
    """Keep only RTH bars: 13:30-20:00 UTC (= 09:30-16:00 EDT)."""
    h = df["time"].dt.hour
    return df[(h >= 13) & (h < 20)].reset_index(drop=True)


def run_variant(days, *, long_only, close_confirm, window_min,
                rr_target=1.0):
    """Simulate one ORB variant over all days. Returns trade list."""
    trades = []
    for day in days:
        if not hasattr(day, "iloc"):
            continue
        if len(day) < 12:            # need enough bars for the range + exit
            continue
        # opening range = first 6 bars of 5m = 30 min (09:30-10:00 ET)
        n_range = 6
        orb = day.iloc[:n_range]
        hi, lo = float(orb["high"].max()), float(orb["low"].min())
        width = hi - lo
        if width <= 0:
            continue
        after = day.iloc[n_range:].reset_index(drop=True)
        if window_min:
            t0 = after["time"].iloc[0]
            cutoff = t0 + pd.Timedelta(minutes=window_min)
            after = after[after["time"] <= cutoff].reset_index(drop=True)
        entered = False
        for i in range(len(after)):
            row = after.iloc[i]
            px_hi, px_lo = float(row["high"]), float(row["low"])
            broke_up = px_hi > hi
            broke_dn = px_lo < lo
            confirmed_up = float(row["close"]) > hi
            up_ok = confirmed_up if close_confirm else broke_up
            dn_ok = broke_dn
            if not entered and up_ok and (not long_only):
                direction, entry, stop, target = 1, hi, lo, hi + rr_target * width
                entered = True; etime = row["time"]; eidx = i
                break
            if not entered and up_ok and long_only:
                direction, entry, stop, target = 1, hi, lo, hi + rr_target * width
                entered = True; etime = row["time"]; eidx = i
                break
            if not entered and (not long_only) and dn_ok:
                direction, entry, stop, target = -1, lo, hi, lo - rr_target * width
                entered = True; etime = row["time"]; eidx = i
                break
        if not entered:
            continue
        # resolve from the bar AFTER entry
        rest = after.iloc[eidx:]
        result = None
        for _, b in rest.iterrows():
            hit_stop = b["low"] <= stop if direction > 0 else b["high"] >= stop
            hit_tgt = b["high"] >= target if direction > 0 else b["low"] <= target
            if hit_stop and hit_tgt:
                result = -1; break          # conservative: stop wins ties
            if hit_stop:
                result = -1; break
            if hit_tgt:
                result = 1; break
        if result is None:                  # end of day flat-close
            close_px = float(after["close"].iloc[-1])
            moved = (close_px - entry) * direction
            result = 1 if moved > 0 else (-1 if moved < 0 else 0)
        trades.append({"time": str(etime), "dir": direction,
                       "win": result == 1, "r": result})
    return trades


def summarize(name, trades):
    if not trades:
        print(f"  {name}: no trades")
        return None
    wins = sum(1 for t in trades if t["r"] > 0)
    wr = wins / len(trades)
    # expectancy in R with 1R risk / RR_TARGET reward
    ev = sum(t["r"] for t in trades) / len(trades)
    print(f"  {name:38} n={len(trades):4d} WR={100*wr:.1f}% EV={ev:+.2f}R")
    return {"name": name, "n": len(trades), "wr": wr, "ev": ev}


def main():
    all_results = {}
    for sym in SYMBOLS:
        try:
            raw = load_5m(sym)
        except FileNotFoundError:
            print(f"{sym}: no 5m data")
            continue
        df = rth_session(raw)
        day_groups = list(df.groupby(df["time"].dt.date))
        days = [d for _, d in day_groups]
        print(f"\n=== {sym} ({len(days)} RTH days) ===")
        res = {}
        res["V1 both-dir 30m"] = summarize(
            "V1 both-dir", run_variant(days, long_only=False,
                                       close_confirm=False, window_min=None))
        res["V2 long-only 30m"] = summarize(
            "V2 long-only", run_variant(days, long_only=True,
                                        close_confirm=False, window_min=None))
        res["V3 long+close-confirm"] = summarize(
            "V3 long+confirm", run_variant(days, long_only=True,
                                           close_confirm=True, window_min=None))
        res["V4 +first-90min"] = summarize(
            "V4 +90min window", run_variant(days, long_only=True,
                                            close_confirm=True, window_min=90))
        all_results[sym] = {k: v for k, v in res.items() if v}

    # aggregate across symbols
    print("\n=== AGGREGATE (5 symbols pooled) ===")
    agg = {}
    for sym, res in all_results.items():
        for k, v in res.items():
            agg.setdefault(k, []).append(v)
    for k, vs in agg.items():
        n = sum(v["n"] for v in vs)
        wr = sum(v["wr"] * v["n"] for v in vs) / max(1, n)
        ev = sum(v["ev"] * v["n"] for v in vs) / max(1, n)
        print(f"  {k:38} n={n:4d} WR={100*wr:.1f}% EV={ev:+.2f}R")
        all_results[k] = {"n": n, "wr": wr, "ev": ev}

    import json
    json.dump(all_results, open(os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp/orb_backtest.json"), "w"),
              indent=2)
    print("saved -> sml_exp/orb_backtest.json")


if __name__ == "__main__":
    main()
