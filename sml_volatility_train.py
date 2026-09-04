#!/usr/bin/env python3
"""sml_volatility_train.py — SML volatility-regime classifier.

Pre-registered (SML_VOLATILITY_PROTOCOL.md): 3-class (CALM/NORMAL/EXPLOSIVE)
classification of next-60-min volatility from the 5m state line. This target
IS learnable (volatility clusters), unlike direction.

Student: Qwen2.5-0.5B + LoRA, 3-way classification, 1650S.
Eval: chronological holdout vs majority-class + persistence baselines.
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
ADAPTER_OUT = os.path.join(HERE, "adapter_volatility")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"

TF = 5
SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
VAL_START = pd.Timestamp("2026-08-20", tz="UTC")
TEST_START = pd.Timestamp("2026-08-24", tz="UTC")
MAX_TRAIN = 8000
EPOCHS = 1
BS = 4
GRAD_ACC = 8
LR = 1e-4
MAX_LEN = 96
CLASSES = ["CALM", "NORMAL", "EXPLOSIVE"]


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


def build_rows():
    rows = []
    for sym in SYMBOLS:
        path = fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_{TF}min.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["datetime"]).rename(
            columns={"datetime": "time"}).sort_values("time").reset_index(drop=True)
        c = df["close"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        for i in range(60, len(df) - 13):
            t = df["time"].iloc[i]
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
            split = ("train" if t < VAL_START else "val" if t < TEST_START
                     else "test")
            # volatility expansion ratio: forward 12-bar return-vol /
            # trailing 20-bar return-vol (apples-to-apples, balanced).
            # thresholds = 25th / 75th percentile (0.68 / 1.29) of the real
            # distribution → ~25/50/25 CALM/NORMAL/EXPLOSIVE split.
            ret = np.diff(np.log(c[max(0, i - 20):i + 13]))
            if len(ret) < 32:
                continue
            tv = ret[0:20].std()     # trailing 20 returns (before i)
            fv = ret[20:32].std()    # forward 12 returns (i..i+11)
            if tv <= 1e-9:
                continue
            ratio = fv / tv
            cls = "CALM" if ratio < 0.68 else ("EXPLOSIVE" if ratio > 1.29
                                               else "NORMAL")
            rows.append({"prompt": state_line(sym, c, h, l, i),
                         "label": cls, "split": split})
    return rows


def main():
    rows = build_rows()
    tr = [r for r in rows if r["split"] == "train"][:MAX_TRAIN]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    print(f"rows: train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    # baselines
    from collections import Counter
    test_labels = [r["label"] for r in te]
    majority = max(Counter(test_labels).values()) / len(test_labels)
    # persistence: previous row's class predicts next (approx via same order)
    pers = 0
    for k in range(1, len(te)):
        if te[k]["label"] == te[k - 1]["label"]:
            pers += 1
    pers /= max(1, len(te) - 1)
    print(f"baselines on test: majority={majority:.3f} persistence={pers:.3f}",
          flush=True)

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
        texts = [f"{r['prompt']} Next hour volatility: CALM, NORMAL, or EXPLOSIVE? Answer with one word.\nAnswer: {r['label']}"
                 for r in rws]
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        labels = enc["input_ids"].clone()
        for i, r in enumerate(rws):
            pre = len(tok(r["prompt"] + " Next hour volatility: CALM, NORMAL, or EXPLOSIVE? Answer with one word.\nAnswer:")["input_ids"])
            labels[i, :pre - 1] = -100
        enc["labels"] = labels
        return {k: v.to(DEVICE) for k, v in enc.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    idx = np.random.RandomState(11).permutation(len(tr))
    model.train()
    opt.zero_grad()
    running, nrun = 0.0, 0
    steps = math.ceil(len(tr) / (BS * GRAD_ACC))
    for bi in range(0, len(idx), BS):
        b = [tr[j] for j in idx[bi:bi + BS]]
        enc = btok(b)
        loss = model(**enc).loss / GRAD_ACC
        loss.backward()
        running += loss.item() * GRAD_ACC
        nrun += 1
        if (bi // BS + 1) % GRAD_ACC == 0 or bi + BS >= len(idx):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
        if nrun % 200 == 0:
            print(f"  step {nrun}/{steps} loss {running/nrun:.4f}", flush=True)
    print(f"done loss {running/max(1,nrun):.4f}", flush=True)
    model.save_pretrained(ADAPTER_OUT)
    tok.save_pretrained(ADAPTER_OUT)
    print(f"adapter saved -> {ADAPTER_OUT}", flush=True)

    # HONEST eval on holdout
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(te), 32):
            chunk = te[i:i + 32]
            enc = tok([r["prompt"] + " Next hour volatility: CALM, NORMAL, or EXPLOSIVE? Answer with one word.\nAnswer:"
                       for r in chunk], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(DEVICE)
            out = model.generate(**enc, max_new_tokens=2, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            toks = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            for j, t in enumerate(toks):
                tu = t.upper()
                pred = ("CALM" if "CALM" in tu else "EXPLOSIVE" if "EXPLOSIVE" in tu
                        else "NORMAL")
                if pred == chunk[j]["label"]:
                    correct += 1
    acc = correct / max(1, len(te))
    print(f"\nHOLDOUT (n={len(te)}): model {acc:.3f} | majority {majority:.3f} | "
          f"persistence {pers:.3f}", flush=True)
    verdict = ("SUCCESS >=80%" if acc >= 0.80 else
               "WEAK 65-80%" if acc >= 0.65 else "DEAD <65%")
    print(f"VERDICT: {verdict}", flush=True)
    json.dump({"model_acc": acc, "majority": majority, "persistence": pers,
               "holdout_n": len(te), "verdict": verdict},
              open(os.path.join(HERE, "volatility_results.json"), "w"))


if __name__ == "__main__":
    main()
