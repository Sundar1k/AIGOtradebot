#!/usr/bin/env python3
"""sml_distill_teacher.py — label 5m market states with the DeepSeek teacher.

Reads 5m CSVs, builds the EXACT state line the SML sees, asks the teacher for
direction + confidence + reason. Writes teacher_labels.jsonl.

Point-in-time: only states up to CUTOFF (2026-08-20) are labeled — the
holdout (Aug 24+) is never touched. Actual next-candle direction is also
stored so we can measure the TEACHER's own holdout-independent accuracy.
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

HERE = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp")
OUT = os.path.join(HERE, "teacher_labels.jsonl")
CUTOFF = pd.Timestamp("2026-08-20", tz="UTC")
TF = 5
SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
MAX_PER_SYMBOL = 600          # 3000 total labeled states
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

# read key from hermes .env (name only, value never printed)
_key = None
for line in open(os.path.join(os.path.expanduser("~"), ".hermes/.env")):
    if line.startswith("DEEPSEEK_API_KEY="):
        _key = line.split("=", 1)[1].strip().strip('"').strip("'")
API_KEY = _key


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


def ask_teacher(state_text):
    import urllib.request
    body = json.dumps({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": (state_text + " Predict the next 5-minute candle: UP or DOWN? "
                        "Answer EXACTLY in this format:\nDIRECTION: UP|DOWN\nCONF: 0.xx\n"
                        "REASON: one short clause")
        }],
        "max_tokens": 60, "temperature": 0,
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    txt = d["choices"][0]["message"]["content"].strip()
    direction = conf = reason = None
    for line in txt.splitlines():
        u = line.upper()
        if u.startswith("DIRECTION"):
            direction = "UP" if "UP" in u else "DOWN"
        elif u.startswith("CONF"):
            try:
                conf = float(line.split(":")[1].strip())
            except Exception:
                conf = 0.5
        elif u.startswith("REASON"):
            reason = line.split(":", 1)[1].strip()[:100]
    direction = direction or ("UP" if "UP" in txt else "DOWN")
    conf = conf if conf is not None else 0.5
    conf = min(0.95, max(0.05, conf))
    return direction, conf, reason or ""


def main():
    tasks = []
    for sym in SYMBOLS:
        path = fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_{TF}min.csv")
        df = pd.read_csv(path, parse_dates=["datetime"]).rename(
            columns={"datetime": "time"}).sort_values("time").reset_index(drop=True)
        c = df["close"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        # deterministic sample: every 4th bar, capped
        idx = list(range(60, len(df) - 1, 4))
        idx = [i for i in idx if df["time"].iloc[i] < CUTOFF][:MAX_PER_SYMBOL]
        for i in idx:
            t = df["time"].iloc[i]
            tasks.append({
                "sym": sym, "i": i,
                "state": state_line(sym, c, h, l, i),
                "actual_next": "UP" if c[i + 1] >= c[i] else "DOWN",
                "time": str(t),
            })
    print(f"states to label: {len(tasks)}", flush=True)

    done = 0
    f = open(OUT, "w")

    def worker(t):
        for attempt in range(4):
            try:
                direction, conf, reason = ask_teacher(t["state"])
                t["teacher_dir"] = direction
                t["teacher_conf"] = conf
                t["teacher_reason"] = reason
                return t
            except Exception as e:
                if attempt == 3:
                    t["teacher_dir"] = None
                    t["teacher_conf"] = 0.5
                    t["teacher_reason"] = f"ERR:{e}"
                    return t
                time.sleep(1.5 * (attempt + 1))

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(worker, tasks):
            f.write(json.dumps(r) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
    f.close()
    print(f"DONE {done} labels -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
