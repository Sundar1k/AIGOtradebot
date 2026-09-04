#!/usr/bin/env python3
"""sml_volatility_eval.py — correct re-eval of the volatility classifier.

The training run's eval was broken (max_new_tokens=2 too short for
"EXPLOSIVE"). This re-loads the saved adapter and evaluates properly:
enough tokens, correct parsing, and a per-class breakdown (the EXPLOSIVE
recall is the number that actually matters).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp")
ADAPTER = os.path.join(HERE, "adapter_volatility")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"
TF = 5
SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
TEST_START = pd.Timestamp("2026-08-24", tz="UTC")
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


def build_test():
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
            if t < TEST_START:
                continue
            ret = np.diff(np.log(c[max(0, i - 20):i + 13]))
            if len(ret) < 32:
                continue
            tv = ret[0:20].std()
            fv = ret[20:32].std()
            if tv <= 1e-9:
                continue
            ratio = fv / tv
            cls = "CALM" if ratio < 0.68 else ("EXPLOSIVE" if ratio > 1.29
                                               else "NORMAL")
            rows.append({"prompt": state_line(sym, c, h, l, i), "label": cls})
    return rows


def main():
    te = build_test()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to(DEVICE)
    model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 32):
            chunk = te[i:i + 32]
            enc = tok([r["prompt"] + " Next hour volatility: CALM, NORMAL, or EXPLOSIVE? Answer with one word.\nAnswer:"
                       for r in chunk], truncation=True, max_length=96,
                      padding=True, return_tensors="pt").to(DEVICE)
            out = model.generate(**enc, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            toks = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            for t in toks:
                tu = t.upper()
                preds.append("CALM" if "CALM" in tu else
                             "EXPLOSIVE" if "EXPLOSIVE" in tu else "NORMAL")

    from collections import Counter
    labels = [r["label"] for r in te]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    acc = correct / len(labels)

    print(f"holdout n={len(labels)}")
    print(f"overall accuracy: {acc:.3f} (majority {max(Counter(labels).values())/len(labels):.3f})")
    print(f"\nper-class (recall = of actual X, how many did it catch):")
    cm = {}
    for cls in CLASSES:
        total = sum(1 for l in labels if l == cls)
        caught = sum(1 for p, l in zip(preds, labels) if l == cls and p == cls)
        pred_count = sum(1 for p in preds if p == cls)
        cm[cls] = (total, caught, pred_count)
        rec = caught / max(1, total)
        print(f"  {cls:10s}: actual {total:5d} | recall {rec:.3f} | predicted {pred_count}")
    # confusion summary
    print("\npredicted distribution:", dict(Counter(preds)))

    json.dump({"overall_acc": acc, "per_class": cm,
               "pred_distribution": dict(Counter(preds))},
              open(os.path.join(HERE, "volatility_eval.json"), "w"), indent=2)
    print("saved -> volatility_eval.json")


if __name__ == "__main__":
    main()
