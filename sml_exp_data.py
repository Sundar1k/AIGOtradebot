#!/usr/bin/env python3
"""sml_exp_data.py — build the pre-registered SML experiment dataset.

5 symbols x 3 timeframes, state-line prompts (live-bot language),
UP/DOWN next-candle labels, chronological splits per protocol.
Writes JSONL: {"prompt":..., "label":"UP"/"DOWN", "symbol":..., "tf":...,
"time":..., "split":"train"/"val"/"test"}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sml_exp")
os.makedirs(OUT, exist_ok=True)

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
TFS = [5, 15, 30]
VAL_START = pd.Timestamp("2026-07-01", tz="UTC")
TEST_START = pd.Timestamp("2026-08-01", tz="UTC")
MAX_ROWS_PER = {"train": 4000, "val": None, "test": None}   # cap train for time


def rsi(closes: np.ndarray, period: int = 14) -> float:
    d = np.diff(closes[-(period + 1):])
    up = d[d > 0].sum() / period
    dn = -d[d < 0].sum() / period
    if dn == 0:
        return 100.0
    rs = up / dn
    return 100 - 100 / (1 + rs)


def stoch(h, l, c, k=14):
    hh = h[-k:].max()
    ll = l[-k:].min()
    if hh == ll:
        return 50.0
    return 100 * (c[-1] - ll) / (hh - ll)


def ema(x: np.ndarray, n: int) -> float:
    return pd.Series(x).ewm(span=n, adjust=False).mean().iloc[-1]


def make_state_line(sym: str, tf: int, df: pd.DataFrame, i: int) -> str:
    c = df["close"].to_numpy(float)[:i + 1]
    h = df["high"].to_numpy(float)[:i + 1]
    l = df["low"].to_numpy(float)[:i + 1]
    r = int(round(rsi(c)))
    e10, e30 = ema(c, 10), ema(c, 30)
    side = "above" if e10 >= e30 else "below"
    st = int(round(stoch(h, l, c)))
    # stochastic direction vs previous bar
    prev_hh = h[-15:-1][-14:].max() if len(h) > 15 else h[:-1][-14:].max() if len(h) > 14 else h.max()
    prev_ll = l[-15:-1][-14:].min() if len(l) > 15 else l[:-1][-14:].min() if len(l) > 14 else l.min()
    prev_c = c[-2]
    prev_st = 100 * (prev_c - prev_ll) / max(1e-9, (prev_hh - prev_ll))
    sdir = "rising" if st >= prev_st else "falling"
    tr = max(h[-1] - l[-1], abs(h[-1] - c[-2]), abs(l[-1] - c[-2]))
    atr = int(round(pd.Series(np.maximum.reduce([
        h[-20:] - l[-20:],
        abs(h[-20:] - c[-21:-1].mean() * 0 + c[-20:]),  # placeholder fixed below
    ])).mean())) if False else int(round(tr))
    return f"{sym} {tf}m. RSI {r}, EMA10 {side} EMA30, stochastic {st} {sdir}, ATR {atr}."


def main():
    total = {"train": 0, "val": 0, "test": 0}
    files = {s: open(os.path.join(OUT, f"{s}.jsonl"), "w") for s in ["all"]}
    fout = files["all"]
    labels_seen = {}
    for sym in SYMBOLS:
        for tf in TFS:
            path = fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_{tf}min.csv")
            if not os.path.exists(path):
                print(f"MISSING {path}")
                continue
            df = pd.read_csv(path, parse_dates=["datetime"]).rename(
                columns={"datetime": "time"})
            df = df.sort_values("time").reset_index(drop=True)
            c = df["close"].to_numpy(float)
            n = len(df)
            kept = 0
            for i in range(60, n - 1):        # need warmup; label uses i+1
                t = df["time"].iloc[i]
                if t.tzinfo is None:
                    t = t.tz_localize("UTC")
                else:
                    t = t.tz_convert("UTC")
                split = ("train" if t < VAL_START
                         else "val" if t < TEST_START else "test")
                lab = "UP" if c[i + 1] >= c[i] else "DOWN"
                key = (split,)
                labels_seen[lab] = labels_seen.get(lab, 0) + 1
                prompt = make_state_line(sym, tf, df.iloc[:i+1], i) \
                    if hasattr(df.iloc[:i+1], 'iloc') else ""
                # simpler: pass numpy views
                row = {"prompt": prompt or _line(sym, tf, df, i),
                       "label": lab, "symbol": sym, "tf": tf,
                       "time": str(t), "split": split}
                # subsample train to keep runtime sane
                if split == "train" and kept % 4 != 0:   # keep 25% of train
                    kept += 1
                    continue
                fout.write(json.dumps(row) + "\n")
                total[split] += 1
            print(f"{sym} {tf}m done (rows so far {total})", flush=True)
    fout.close()
    json.dump({"total": total, "labels": labels_seen},
              open(os.path.join(OUT, "meta.json"), "w"))
    print("DONE", total, labels_seen)


def _line(sym, tf, df, i):
    c = df["close"].to_numpy(float)[max(0, i-59):i+1]
    h = df["high"].to_numpy(float)[max(0, i-59):i+1]
    l = df["low"].to_numpy(float)[max(0, i-59):i+1]
    r = int(round(rsi(c)))
    e10, e30 = ema(c, 10), ema(c, 30)
    side = "above" if e10 >= e30 else "below"
    st = int(round(stoch(h, l, c)))
    if len(c) >= 16:
        hh, ll = h[-15:-1].max(), l[-15:-1].min()
    else:
        hh, ll = h.max(), l.min()
    prev_st = 100 * (c[-2] - ll) / max(1e-9, hh - ll) if len(c) >= 2 and hh > ll else 50.0
    sdir = "rising" if st >= prev_st else "falling"
    tr = max(h[-1] - l[-1], abs(h[-1] - c[-2]), abs(l[-1] - c[-2])) if len(c) >= 2 else h[-1]-l[-1]
    atr = int(round(tr))
    return f"{sym} {tf}m. RSI {r}, EMA10 {side} EMA30, stochastic {st} {sdir}, ATR {atr}. Next candle UP or DOWN?"


if __name__ == "__main__":
    main()
