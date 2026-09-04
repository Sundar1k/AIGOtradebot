#!/usr/bin/env python3
"""sml_volatility_train_v2.py — collapse-repaired volatility classifier.

Per SML_VOLATILITY_PROTOCOL_V2.md (locked 2026-08-28). Same target, same
features, same splits as v1; ONLY changes are (1) class-weighted loss and
(2) direct logit eval over the three label tokens instead of greedy decode.

This is the LAST volatility attempt per protocol. Report honestly.
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
ADAPTER_OUT = os.path.join(HERE, "adapter_volatility_v2")
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
    tr = max(hh[-1] - ll[-1], abs(hh[-1] - cc[-2]), abs(ll[-1] - cc[-2]))
    return (f"{sym} {TF}m. RSI {int(round(rsi))}, EMA10 {side} EMA30, "
            f"stochastic {int(round(st))}, ATR {int(round(tr))}.")


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
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            split = ("train" if t < VAL_START else "val" if t < TEST_START else "test")
            ret = np.diff(np.log(c[max(0, i - 20):i + 13]))
            if len(ret) < 32:
                continue
            tv = ret[0:20].std()
            fv = ret[20:32].std()
            if tv <= 1e-9:
                continue
            ratio = fv / tv
            cls = "CALM" if ratio < 0.68 else ("EXPLOSIVE" if ratio > 1.29 else "NORMAL")
            rows.append({"prompt": state_line(sym, c, h, l, i),
                         "label": cls, "split": split})
    return rows


def build_class_weights(rows):
    from collections import Counter
    cnt = Counter(r["label"] for r in rows)
    total = sum(cnt.values())
    w = {k: total / (len(cnt) * v) for k, v in cnt.items()}
    print(f"class counts: {dict(cnt)} -> weights {w}", flush=True)
    return {k: torch.tensor(w[k], dtype=torch.float32) for k in w}


def main():
    rows = build_rows()
    tr = [r for r in rows if r["split"] == "train"][:MAX_TRAIN]
    te = [r for r in rows if r["split"] == "test"]
    print(f"rows: train={len(tr)} test={len(te)}", flush=True)

    from collections import Counter
    test_labels = [r["label"] for r in te]
    majority = max(Counter(test_labels).values()) / len(test_labels)
    pers = sum(1 for k in range(1, len(te)) if te[k]["label"] == te[k - 1]["label"]) / max(1, len(te) - 1)
    print(f"baselines on test: majority={majority:.3f} persistence={pers:.3f}", flush=True)

    cw = build_class_weights(tr)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEVICE)
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    Q = " Next hour volatility: CALM, NORMAL, or EXPLOSIVE? Answer with one word.\nAnswer:"

    def btok(rws):
        texts = [f"{r['prompt']}{Q} {r['label']}" for r in rws]
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        labels = enc["input_ids"].clone()
        for i, r in enumerate(rws):
            pre = len(tok(r["prompt"] + Q)["input_ids"])
            labels[i, :pre - 1] = -100
        enc["labels"] = labels
        return {k: v.to(DEVICE) for k, v in enc.items()}

    # label-token ids for direct logit eval
    label_ids = {c: tok.encode(f" {c}", add_special_tokens=False)[0] for c in CLASSES}
    print(f"label token ids: {label_ids}", flush=True)

    # class-weight the loss via per-sample weight on label positions
    def loss_weighted(enc, rws):
        W = torch.ones(enc["labels"].shape, dtype=torch.float32, device=DEVICE)
        for i, r in enumerate(rws):
            w = cw[r["label"]]
            mask = enc["labels"][i] != -100
            W[i][mask] = w
        out = model(**enc)
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = enc["labels"][..., 1:].contiguous()
        lf = torch.nn.CrossEntropyLoss(reduction="none")
        loss = lf(shift_logits.view(-1, shift_logits.size(-1)),
                  shift_labels.view(-1))
        loss = loss.view(shift_labels.shape)
        wshift = W[..., 1:]
        mask = shift_labels != -100
        loss = (loss * wshift)[mask].mean()
        return loss

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    idx = np.random.RandomState(11).permutation(len(tr))
    model.train()
    opt.zero_grad()
    running, nrun = 0.0, 0
    for bi in range(0, len(idx), BS):
        b = [tr[j] for j in idx[bi:bi + BS]]
        enc = btok(b)
        loss = loss_weighted(enc, b) / GRAD_ACC
        loss.backward()
        running += loss.item() * GRAD_ACC
        nrun += 1
        if (bi // BS + 1) % GRAD_ACC == 0 or bi + BS >= len(idx):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
        if nrun % 200 == 0:
            print(f"  step {nrun} loss {running/nrun:.4f}", flush=True)
    print(f"done loss {running/max(1,nrun):.4f}", flush=True)
    model.save_pretrained(ADAPTER_OUT)
    tok.save_pretrained(ADAPTER_OUT)
    print(f"adapter saved -> {ADAPTER_OUT}", flush=True)

    # HONEST eval: direct logit over the 3 label tokens
    model.eval()
    correct = 0
    per_class = {c: [0, 0] for c in CLASSES}   # [actual, predicted-correct]
    pred_counts = {c: 0 for c in CLASSES}
    with torch.no_grad():
        for i in range(0, len(te), 32):
            chunk = te[i:i + 32]
            enc = tok([r["prompt"] + Q for r in chunk], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            # logit of the token right after the LAST REAL token (not pad);
            # attention mask gives the true end position per row.
            logits = out.logits
            last_pos = enc["attention_mask"].sum(dim=1) - 1  # last non-pad index
            last = logits[torch.arange(len(chunk)), last_pos, :]
            for j, r in enumerate(chunk):
                scores = {c: float(last[j, label_ids[c]].item()) for c in CLASSES}
                pred = max(scores, key=scores.get)
                per_class[r["label"]][0] += 1
                pred_counts[pred] += 1
                if pred == r["label"]:
                    correct += 1
                    per_class[r["label"]][1] += 1
    acc = correct / max(1, len(te))
    print(f"\nHOLDOUT (n={len(te)}): model {acc:.3f} | majority {majority:.3f} | "
          f"persistence {pers:.3f}", flush=True)
    for c in CLASSES:
        a, p = per_class[c]
        print(f"  {c:9s}: actual {a:5d} | correct {p:5d} | recall {p/max(1,a):.3f}", flush=True)
    print(f"  predicted distribution: {pred_counts}", flush=True)
    verdict = ("SUCCESS" if acc >= 0.80 else "WEAK" if acc >= 0.65 else "DEAD")
    print(f"VERDICT: {verdict}", flush=True)
    json.dump({"model_acc": acc, "majority": majority, "persistence": pers,
               "holdout_n": len(te), "per_class": {c: per_class[c] for c in CLASSES},
               "pred_distribution": pred_counts, "verdict": verdict},
              open(os.path.join(HERE, "volatility_results_v2.json"), "w"), indent=2)


if __name__ == "__main__":
    main()