import os
#!/usr/bin/env python3
"""value_edge_failure.py — Creamer value-edge failure selector (spec: specs/value-edge-failure/SPEC.md).

Point-in-time only. Prior RTH session builds the volume profile; today's 09:30-12:00 ET
window is scanned for penetration -> failure -> retry-fails-higher -> flip.
Simulation: stop-first on collisions, target variants A (POC) and B (prior swing),
session-end exit at 16:00 ET close. Nothing touches live config.
"""
import argparse, json, os, math
from collections import defaultdict
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TICKS = {"NQ": 0.25, "ES": 0.25, "RTY": 0.05, "YM": 1.0, "GC": 0.10}
PEN_ATR = 0.30          # penetration depth in prior-day ATR14 units
FAIL_BARS = 12          # max bars from first penetration to failure close
RETRY_BARS = 20         # max bars from failure close to valid retry+flip
VOL_FLOOR = 0.5         # signal bar volume >= 0.5 * same-hour median (prior 30 sessions)
SMA_N = 14
W0, W1 = "09:30", "12:00"   # entry window ET
R0, R1 = "09:30", "16:00"   # RTH session ET


def load_bars(sym):
    df = pd.read_csv(f"{BASE}/data/{sym}_3min.csv",
                     usecols=["datetime", "open", "high", "low", "close", "volume"])
    df["utc"] = pd.to_datetime(df["datetime"], utc=True)
    df["et"] = df["utc"].dt.tz_convert("America/New_York")
    return df


def build_day_profile(h, l, v, tick):
    binw = max(tick * 4.0, 0.25)
    lo = math.floor(float(min(l)) / binw) * binw
    hi = math.ceil(float(max(h)) / binw) * binw
    nbin = max(1, int(round((hi - lo) / binw)))
    edges = lo + np.arange(nbin + 1) * binw
    mid = (np.asarray(h, float) + np.asarray(l, float)) / 2.0
    w = np.asarray(v, float)
    hist, _ = np.histogram(mid, bins=edges, weights=w)
    if hist.sum() <= 0:
        return None
    total = hist.sum()
    poc_i = int(np.argmax(hist))
    centers = (edges[:-1] + edges[1:]) / 2.0
    acc = hist[poc_i]
    left = right = poc_i
    while acc < 0.70 * total:
        gl = hist[left - 1] if left > 0 else -1.0
        gr = hist[right + 1] if right < nbin - 1 else -1.0
        if gl < 0 and gr < 0:
            break
        if gl >= gr:
            left -= 1; acc += hist[left]
        else:
            right += 1; acc += hist[right]
    return {"poc": float(centers[poc_i]), "va_low": float(edges[left]),
            "va_high": float(edges[right + 1])}


def scan_symbol(sym, verbose=False):
    tick = TICKS.get(sym, 0.25)
    df = load_bars(sym)
    et = df["et"]
    t0, t1 = pd.to_datetime(R0).time(), pd.to_datetime(R1).time()
    rth = (et.dt.time >= t0) & (et.dt.time < t1)
    prev_c = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()))
    atr = tr.rolling(SMA_N).mean().to_numpy(float)

    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)

    rth_idx = np.where(rth.to_numpy())[0]
    by_date = {}
    for i in rth_idx:
        by_date.setdefault(et.iloc[i].date(), []).append(i)
    dates = sorted(by_date.keys())

    # per-(date,hour) volume arrays for the participation filter
    hour_hist = []   # list of (date, {hour: np.array})
    signals = []

    for di, d in enumerate(dates):
        idxs = by_date[d]
        hours = {}
        for i in idxs:
            hours.setdefault(et.iloc[i].strftime("%H"), []).append(i)
        hour_hist.append((d, {hk: v[hi] for hk, hi in hours.items()}))

        prev = None
        if di >= 1:
            pidxs = by_date[dates[di - 1]]
            if len(pidxs) >= 10:
                prof = build_day_profile(h[pidxs], l[pidxs], v[pidxs], tick)
                a = atr[pidxs[-1]]
                if prof and not np.isnan(a) and a > 0:
                    prev = {**prof, "atr": float(a),
                            "close": float(c[pidxs[-1]]),
                            "high": float(np.max(h[pidxs])), "low": float(np.min(l[pidxs]))}
        if prev is None:
            continue

        long_ok = prev["close"] > prev["poc"]
        short_ok = prev["close"] < prev["poc"]
        if not (long_ok or short_ok):
            continue

        pen = PEN_ATR * prev["atr"]
        win = [i for i in idxs if W0 <= et.iloc[i].strftime("%H:%M") < W1]
        if not win:
            continue
        last_rth_i = idxs[-1]

        def vol_floor_ok(i):
            hh = et.iloc[i].strftime("%H")
            arrs = []
            for dd, hhmap in hour_hist:
                if dd >= d:
                    break
                if hh in hhmap:
                    arrs.append(hhmap[hh])
            if not arrs:
                return True
            med = float(np.median(np.concatenate(arrs[-30:])))
            return med <= 0 or v[i] >= VOL_FLOOR * med

        for side in (["long"] if long_ok else []) + (["short"] if short_ok else []):
            if side == "long":
                edge = prev["va_low"]
            else:
                edge = prev["va_high"]
            L1 = None; failed = False; fail_i = None
            retry_low = None; retry_high = None
            ctr = {"attempt": 0, "fail": 0, "retry": 0, "flip": 0, "invalid": 0, "emit": 0}
            for pos, i in enumerate(win):
                if side == "long":
                    attempt = l[i] < edge - pen
                    invalidation = (L1 is not None) and (l[i] <= L1)
                    failed_close = c[i] >= edge
                    retry_attempt = failed and (retry_low is None) and \
                        (l[i] < edge) and (l[i] > L1)
                    flip = (retry_low is not None) and (c[i] >= max(edge, retry_high))
                else:
                    attempt = h[i] > edge + pen
                    invalidation = (L1 is not None) and (h[i] >= L1)
                    failed_close = c[i] <= edge
                    retry_attempt = failed and (retry_low is None) and \
                        (h[i] > edge) and (h[i] < L1)
                    flip = (retry_low is not None) and (c[i] <= min(edge, retry_high))

                if not failed and L1 is not None and invalidation:
                    # second attempt made a NEW extreme -> restart the cycle
                    L1 = None; failed = False; fail_i = None
                    ctr["invalid"] += 1
                    continue
                if L1 is None and attempt:
                    L1 = l[i] if side == "long" else h[i]
                    ctr["attempt"] += 1
                elif L1 is not None and not failed:
                    L1 = min(L1, l[i]) if side == "long" else max(L1, h[i])
                    if failed_close:
                        failed = True; fail_i = i
                        ctr["fail"] += 1
                elif failed and fail_i is not None and (i - fail_i) <= RETRY_BARS:
                    if retry_low is None and retry_attempt:
                        retry_low = l[i] if side == "long" else h[i]
                        retry_high = h[i] if side == "long" else l[i]
                        ctr["retry"] += 1
                    if flip and vol_floor_ok(i):
                        ctr["flip"] += 1
                        E = float(c[i])
                        if side == "long":
                            stop = min(L1, retry_low) - 0.10 * prev["atr"]
                        else:
                            stop = max(L1, retry_high if False else retry_low) + 0.10 * prev["atr"]
                        if side == "long":
                            if not (stop < E):
                                break
                            tgts = [("A_poc", prev["poc"]), ("B_swing", prev["high"])]
                        else:
                            if not (stop > E):
                                break
                            tgts = [("A_poc", prev["poc"]), ("B_swing", prev["low"])]
                        for tname, tgt in tgts:
                            if (side == "long" and tgt <= E) or (side == "short" and tgt >= E):
                                continue
                            r = simulate(i, idxs, E, stop, tgt, last_rth_i, o, h, l, c, side)
                            if r is None:
                                continue
                            signals.append({
                                "ts": str(et.iloc[i]), "symbol": sym, "side": side,
                                "entry": E, "stop": stop, "target": tgt,
                                "L1": float(L1), "retry_low": float(retry_low),
                                "va_low": prev["va_low"], "va_high": prev["va_high"],
                                "poc": prev["poc"], "atr": prev["atr"],
                                "variant": tname, "outcome_r": r,
                                "exit": "stop" if r <= -0.999 else ("target" if r > 0 else "session"),
                            })
                        break  # one trade per side per day
    return signals


def simulate(idx, day_idxs, E, stop, target, last_rth_i, o, h, l, c, side="long"):
    if side == "long":
        risk = E - stop
    else:
        risk = stop - E
    if risk <= 0:
        return None
    for j in day_idxs:
        if j <= idx:
            continue
        if side == "long":
            if l[j] <= stop:
                return -1.0
            if h[j] >= target:
                return (target - E) / risk
        else:
            if h[j] >= stop:
                return -1.0
            if l[j] <= target:
                return (E - target) / risk
        if j >= last_rth_i:
            break
    close_r = (c[last_rth_i] - E) / risk if side == "long" else (E - c[last_rth_i]) / risk
    return close_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["NQ"])
    args = ap.parse_args()
    all_sig = []
    for sym in args.symbols:
        sigs = scan_symbol(sym)
        all_sig += sigs
        print(f"{sym}: {len(sigs)} signals")
    out = os.path.join(BASE, "specs", "value-edge-failure", "signals.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for s in all_sig:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(all_sig)} signals -> {out}")


if __name__ == "__main__":
    main()
