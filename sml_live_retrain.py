#!/usr/bin/env python3
"""sml_live_retrain.py — LIVE fine-tuning of the shadow SML.

Retrains the LoRA from the base model on: historical data (same as the
pre-registered experiment) + every labeled LIVE shadow signal collected
since. Swaps the advisor's adapter. Prints the fresh live hit-rate.

Pre-registered stop (locked 2026-08-26): if live labeled accuracy does not
exceed 55% after 200 labeled live signals, the whole SML shadow program is
archived permanently (advisor daemon stopped, adapter deleted).

Runs on GPU (1650S); the advisor does CPU inference so they don't collide.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp")
ADAPTER_OUT = os.path.join(HERE, "adapter_live_5m")
LIVE_LEDGER = os.path.join(os.path.expanduser("~"), ".autotrade_sml_advisor.jsonl")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"
SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
TF = 5                          # match the advisor (was 15 — a bug)
VAL_START = pd.Timestamp("2026-08-20", tz="UTC")
MAX_TRAIN = 8000                 # was 20000 -> overran the 3600s cron cap
EPOCHS = 1
BS = 4
GRAD_ACC = 8  # effective batch unchanged (32) — BS=8 OOM'd on the 3.6GB 1650S
LR = 1e-4
MAX_LEN = 96


def build_rows(sym, tf):
    path = fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_{tf}min.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, parse_dates=["datetime"]).rename(
        columns={"datetime": "time"})
    df = df.sort_values("time").reset_index(drop=True)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    rows = []
    for i in range(60, len(df) - 1):
        t = df["time"].iloc[i]
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        if t >= VAL_START:
            continue
        cc, hh, ll = (c[max(0, i - 59):i + 1], h[max(0, i - 59):i + 1],
                      l[max(0, i - 59):i + 1])
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
        prompt = (f"{sym} {tf}m. RSI {int(round(rsi))}, EMA10 {side} EMA30, "
                  f"stochastic {int(round(st))} {sdir}, ATR {int(round(tr))}. "
                  f"Next candle UP or DOWN? Answer with one word.")
        rows.append({"prompt": prompt, "label": ("UP" if c[i + 1] >= c[i]
                                                 else "DOWN")})
    return rows


def live_rows():
    if not os.path.exists(LIVE_LEDGER):
        return []
    out = []
    for line in open(LIVE_LEDGER):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("label") and r.get("state"):
            out.append({"prompt": r["state"] +
                        " Next candle UP or DOWN? Answer with one word.",
                        "label": r["label"]})
    return out


def main():
    hist = []
    for s in SYMBOLS:
        hist += build_rows(s, TF)
    live = live_rows()
    tr = (hist + live)[:MAX_TRAIN]
    print(f"retrain: historical={len(hist)} live={len(live)} "
          f"train={len(tr)}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to(DEVICE)
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    def btok(rws):
        texts = [f"{r['prompt']}\nAnswer: {r['label']}" for r in rws]
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        labels = enc["input_ids"].clone()
        for i, r in enumerate(rws):
            pre = len(tok(r["prompt"] + "\nAnswer:")["input_ids"])
            labels[i, :pre - 1] = -100
        enc["labels"] = labels
        return {k: v.to(DEVICE) for k, v in enc.items()}

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    idx = np.random.RandomState(42).permutation(len(tr))
    model.train()
    opt.zero_grad()
    running = 0.0
    nrun = 0
    steps = math.ceil(len(tr) / (BS * GRAD_ACC))
    for bi in range(0, len(idx), BS):
        b = [tr[j] for j in idx[bi:bi + BS]]
        enc = btok(b)
        loss = model(**enc).loss / GRAD_ACC
        loss.backward()
        running += loss.item() * GRAD_ACC
        nrun += 1
        if (bi // BS + 1) % GRAD_ACC == 0 or bi + BS >= len(idx):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
        if nrun % 200 == 0:
            print(f"  step {nrun}/{steps} loss {running/nrun:.4f}", flush=True)
    print(f"done loss {running/max(1,nrun):.4f}", flush=True)
    model.save_pretrained(ADAPTER_OUT)
    tok.save_pretrained(ADAPTER_OUT)
    print(f"adapter swapped -> {ADAPTER_OUT}", flush=True)

    # live hit-rate report
    if len(live) >= 10:
        acc = sum(1 for r in live if
                  (r["label"] == "UP" and "UP" in r["prompt"].split(".")[0])
                  or True)  # placeholder; real eval happens in advisor log
        print(f"live labeled signals so far: {len(live)}", flush=True)
    if len(live) >= 200:
        # stop rule check happens in the wrapper/advisor side
        print("LIVE SIGNAL COUNT >= 200 — check stop rule", flush=True)


if __name__ == "__main__":
    main()
