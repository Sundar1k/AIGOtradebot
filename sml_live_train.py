#!/usr/bin/env python3
"""sml_live_train.py — retrain Qwen2.5-0.5B LoRA for the LIVE SHADOW advisor.
Same data/geometry as the pre-registered experiment (15m, all 5 symbols,
chronological splits) but shorter (15k rows, 1 epoch) and SAVES the adapter.
The live advisor is a measuring instrument only — it never places orders.
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
ADAPTER_OUT = os.path.join(HERE, "adapter_live")

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
TF = 15
VAL_START = pd.Timestamp("2026-07-01", tz="UTC")
TEST_START = pd.Timestamp("2026-08-01", tz="UTC")
MAX_TRAIN = 15000
EPOCHS = 1
BS = 8
GRAD_ACC = 4
LR = 1e-4
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"
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
        split = ("train" if t < VAL_START else "val" if t < TEST_START
                 else "test")
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
        label = "UP" if c[i + 1] >= c[i] else "DOWN"
        rows.append({"prompt": prompt, "label": label, "split": split})
    return rows


def main():
    rows = []
    for s in SYMBOLS:
        rows += build_rows(s, TF)
    tr = [r for r in rows if r["split"] == "train"][:MAX_TRAIN]
    te = [r for r in rows if r["split"] == "test"]
    print(f"train={len(tr)} test={len(te)}", flush=True)

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
    model.print_trainable_parameters()

    def batch_tokenize(rws):
        texts = [f"{r['prompt']}\nAnswer: {r['label']}" for r in rws]
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        labels = enc["input_ids"].clone()
        for i, r in enumerate(rws):
            pre = len(tok(r["prompt"] + "\nAnswer:")["input_ids"])
            labels[i, :pre - 1] = -100
        enc["labels"] = labels
        return {k: v.to(DEVICE) for k, v in enc.items()}

    def evaluate(rws):
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(rws), 32):
                chunk = rws[i:i + 32]
                enc = tok([r["prompt"] + "\nAnswer:" for r in chunk],
                          truncation=True, max_length=MAX_LEN, padding=True,
                          return_tensors="pt").to(DEVICE)
                out = model.generate(**enc, max_new_tokens=1, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                toks = tok.batch_decode(out[:, -1].unsqueeze(1))
                for j, t in enumerate(toks):
                    p = ("UP" if "UP" in t.upper()
                         and "DOWN" not in t.upper() else "DOWN")
                    if p == chunk[j]["label"]:
                        correct += 1
        model.train()
        return correct / max(1, len(rws))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    idx = np.random.RandomState(42).permutation(len(tr))
    steps = math.ceil(len(tr) / (BS * GRAD_ACC))
    model.train()
    opt.zero_grad()
    for ep in range(EPOCHS):
        running = 0.0
        nrun = 0
        for bi in range(0, len(idx), BS):
            b = [tr[j] for j in idx[bi:bi + BS]]
            enc = batch_tokenize(b)
            out = model(**enc)
            loss = out.loss / GRAD_ACC
            loss.backward()
            running += loss.item() * GRAD_ACC
            nrun += 1
            if (bi // BS + 1) % GRAD_ACC == 0 or bi + BS >= len(idx):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
            if nrun % 200 == 0:
                print(f"  step {nrun}/{steps} loss {running/nrun:.4f}",
                      flush=True)
        print(f"epoch {ep+1} loss {running/max(1,nrun):.4f}", flush=True)
    model.save_pretrained(ADAPTER_OUT)
    tok.save_pretrained(ADAPTER_OUT)
    print(f"adapter saved -> {ADAPTER_OUT}", flush=True)

    tacc = evaluate(te[:1500])
    print(f"TEST ACC (sub 1500): {tacc:.4f} | always-up {sum(1 for r in te if r['label']=='UP')/len(te):.4f}",
          flush=True)
    json.dump({"test_acc": tacc, "test_n": min(1500, len(te)),
               "note": "live shadow advisor training"},
              open(os.path.join(HERE, "live_train_results.json"), "w"))


if __name__ == "__main__":
    main()
