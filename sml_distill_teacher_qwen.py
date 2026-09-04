#!/usr/bin/env python3
"""sml_distill_teacher_qwen.py — Qwen-7B TEACHER labels 5m states for KD.

Uses the RUNNING veto service (Qwen-7B on the 1070) as the teacher via its
/decide_batch HTTP endpoint — this sidesteps the sm_61 "no kernel image"
problem entirely (the veto service already loads/executes the model
correctly; we never touch the GPU from this script).

Action map: BUY -> UP, SELL -> DOWN, NO TRADE -> skip (no label).
Point-in-time: only states < CUTOFF labeled; holdout never touched.
Runs on CPU (just HTTP + pandas) — no CUDA_VISIBLE_DEVICES needed.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp")
OUT = os.path.join(HERE, "teacher_labels_qwen.jsonl")
CUTOFF = pd.Timestamp("2026-08-20", tz="UTC")
TF = 5
SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
MAX_PER_SYMBOL = 400
BATCH = 32
VETO_URL = "http://127.0.0.1:8765/decide_batch"


def state_line(sym, c, h, l, i):
    cc = c[max(0, i - 59):i + 1]
    hh = h[max(0, i - 59):i + 1]
    ll = l[max(0, i - 59):i + 1]
    d = np.diff(cc[-15:])
    up = d[d > 0].sum() / 14
    dn = -d[d < 0].sum() / 14
    rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
    e10 = pd.Series(cc).ewm(span=10, adjust=False).mean().iloc[-1]
    e30 = pd.Series(cc).ewm(span=30, adjust=False).mean().iloc[-1]
    side = "above" if e10 >= e30 else "below"
    hhk, llk = hh[-14:].max(), ll[-14:].min()
    st = 100 * (cc[-1] - llk) / max(1e-9, hhk - llk)
    prev_hh, prev_ll = hh[-15:-1][-14:].max(), ll[-15:-1][-14:].min()
    prev_st = 100 * (cc[-2] - prev_ll) / max(1e-9, prev_hh - prev_ll)
    sdir = "rising" if st >= prev_st else "falling"
    tr = max(hh[-1] - ll[-1], abs(hh[-1] - cc[-2]), abs(ll[-1] - cc[-2]))
    return (f"{sym} {TF}m. RSI {int(round(rsi))}, EMA10 {side} EMA30, "
            f"stochastic {int(round(st))} {sdir}, ATR {int(round(tr))}.")


def ask_batch(states):
    body = json.dumps({"texts": states}).encode()
    req = urllib.request.Request(VETO_URL, data=body, headers={
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d.get("results", [])


def main():
    tasks = []
    for sym in SYMBOLS:
        path = fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_{TF}min.csv")
        df = pd.read_csv(path, parse_dates=["datetime"]).rename(
            columns={"datetime": "time"}).sort_values("time").reset_index(drop=True)
        c = df["close"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        idx = [i for i in range(60, len(df) - 1, 4)
               if df["time"].iloc[i] < CUTOFF][:MAX_PER_SYMBOL]
        for i in idx:
            t = df["time"].iloc[i]
            tasks.append({
                "sym": sym, "state": state_line(sym, c, h, l, i),
                "actual_next": "UP" if c[i + 1] >= c[i] else "DOWN",
                "time": str(t),
            })
    print(f"states to label via veto service: {len(tasks)}", flush=True)

    f = open(OUT, "w")
    done = 0
    t0 = time.time()
    for bi in range(0, len(tasks), BATCH):
        chunk = tasks[bi:bi + BATCH]
        try:
            results = ask_batch([t["state"] for t in chunk])
        except Exception as e:
            print(f"  batch error @{bi}: {e}", flush=True)
            time.sleep(5)
            continue
        for t, r in zip(chunk, results):
            a = r.get("action", "NO TRADE")
            if a == "BUY":
                t["teacher_dir"] = "UP"; t["teacher_conf"] = 0.62
            elif a == "SELL":
                t["teacher_dir"] = "DOWN"; t["teacher_conf"] = 0.62
            else:
                t["teacher_dir"] = None; t["teacher_conf"] = 0.5
            t["teacher_reason"] = r.get("reason", "")[:100]
            f.write(json.dumps(t) + "\n")
            done += 1
        if (bi // BATCH) % 5 == 0:
            print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"DONE {done} labels -> {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
