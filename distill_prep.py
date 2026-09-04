#!/usr/bin/env python3
"""distill_prep.py — build the Qwen 1.5B distillation dataset.

Teacher labels come from the LIVE 7B veto sidecar (:8765 /decide_batch) on
state lines built with the EXACT v1 training format (build_state_line).
Resumable: states are staged first, labels appended in chunks — a restart
skips already-labeled states. GPU is free (veto.service holds the model).

Run:  nohup-style background via Hermes terminal; check finetune/distill/
Usage: .venv/bin/python distill_prep.py [--n 1200]
"""
import json
import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import indicators as ind
import signals

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
OUT = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/finetune/distill")
VETO_URL = "http://127.0.0.1:8765/decide_batch"
CHUNK = 10                # batched GPU generate (v2); 10 x ~8s per request


def build_states(symbol: str, n_per_symbol: int) -> list:
    """State lines for n evenly-spread bars per symbol (exact v1 format)."""
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{symbol}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.rename(columns={"time": "datetime"})
    sc = signals.compute_scores(df)
    c = df["close"].to_numpy(float)
    ema_f = pd.Series(c).ewm(span=10, adjust=False).mean().to_numpy()
    ema_s = pd.Series(c).ewm(span=30, adjust=False).mean().to_numpy()
    kk, dd = signals.stochastic(df)
    kk = kk.to_numpy(float)
    dd = dd.to_numpy(float)
    n = len(df)
    idx = np.linspace(200, n - 1, n_per_symbol).astype(int)
    rng = np.random.default_rng(42)
    idx = np.unique(np.concatenate([idx, rng.integers(200, n - 1, n_per_symbol)]))
    states = []
    for i in idx:
        row = sc.iloc[i]
        if not (np.isfinite(row["rsi"]) and np.isfinite(row["atr"])
                and np.isfinite(row["score"])):
            continue
        ema_dir = "above" if ema_f[i] > ema_s[i] else "below"
        stoch = float(kk[i])
        if not np.isfinite(stoch):
            continue
        stoch_trend = "rising" if kk[i] >= dd[i] else "falling"
        line = (f"{config.base_symbol(symbol)} 3m. RSI {row['rsi']:.0f}, "
                f"EMA10 {ema_dir} EMA30, stochastic {stoch:.0f} {stoch_trend}, "
                f"ATR {row['atr']:.0f}. Score {int(row['score']):+d}."
                f" Answer with the trade action and one reason line.")
        states.append(line)
    return states


def post_batch(texts):
    req = urllib.request.Request(
        VETO_URL, data=json.dumps({"texts": texts}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main():
    n_per = int(os.environ.get("N_PER_SYMBOL", "240"))
    os.makedirs(OUT, exist_ok=True)

    states_path = os.path.join(OUT, "states.jsonl")
    labels_path = os.path.join(OUT, "labels.jsonl")

    # STAGE 1 — stage states (idempotent)
    if not os.path.exists(states_path):
        with open(states_path, "w") as f:
            for sym in SYMBOLS:
                for s in build_states(sym, n_per):
                    f.write(json.dumps({"symbol": sym, "text": s}) + "\n")
        print(f"staged states -> {states_path}", flush=True)
    with open(states_path) as f:
        states = [json.loads(l) for l in f if l.strip()]
    print(f"total states: {len(states)}", flush=True)

    # STAGE 2 — teacher labels via live sidecar (resumable)
    done = set()
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            for l in f:
                if l.strip():
                    done.add(json.loads(l)["text"])
    todo = [s for s in states if s["text"] not in done]
    print(f"already labeled: {len(done)} · to do: {len(todo)} "
          f"(~{len(todo)*17/60:.0f} min GPU)", flush=True)

    with open(labels_path, "a") as out:
        failed = False
        for k in range(0, len(todo), CHUNK):
            chunk = todo[k:k + CHUNK]
            texts = [c["text"] for c in chunk]
            try:
                d = post_batch(texts)
                for c, res in zip(chunk, d.get("results", [])):
                    rec = {"text": c["text"], "symbol": c["symbol"],
                           "action": res.get("action", "NO TRADE"),
                           "reason": res.get("reason", "")}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    done.add(c["text"])
            except Exception as e:
                print(f"batch failed at {k}: {type(e).__name__}: {e} — "
                      f"resume by re-running", flush=True)
                failed = True
                break
            if (k // CHUNK) % 10 == 0:
                print(f"labeled {len(done)}/{len(states)} "
                      f"({100*len(done)/max(1,len(states)):.0f}%)", flush=True)
        if failed:
            # honest exit: a partial run must NOT report success
            print("PARTIAL RUN — exiting with code 1 (resume to continue)", flush=True)
            return 1

    # STAGE 3 — split train/val + stats
    rows = [json.loads(l) for l in open(labels_path) if l.strip()]
    if rows:
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(rows))
        n_val = max(1, int(0.25 * len(rows)))
        val_idx = set(perm[:n_val].tolist())
        with open(os.path.join(OUT, "train.jsonl"), "w") as tf, \
             open(os.path.join(OUT, "val.jsonl"), "w") as vf:
            for j, r in enumerate(rows):
                target = vf if j in val_idx else tf
                comp = f"{r['action']} {r['reason']}".strip()
                target.write(json.dumps({"text": r["text"], "completion": comp}) + "\n")
        from collections import Counter
        acts = Counter(r["action"] for r in rows)
        print("=" * 60, flush=True)
        print(f"DATASET READY: {len(rows)} labeled | train/val split saved", flush=True)
        print(f"action distribution: {dict(acts)}", flush=True)
    else:
        print("no labels yet — rerun when the sidecar is up", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
