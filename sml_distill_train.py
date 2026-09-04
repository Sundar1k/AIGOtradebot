#!/usr/bin/env python3
"""sml_distill_train.py — distill the DeepSeek teacher into the 0.5B student.

Training losses (textbook KD):
  L = alpha * CE(hard teacher direction) + (1-alpha) * T^2 * KL(soft targets)
Soft target = teacher's stated confidence mapped to a P(UP) distribution
over the answer tokens.

Student: Qwen2.5-0.5B-Instruct + LoRA (same recipe as before, GPU 1650S).
Output adapter: sml_exp/adapter_distill

Honest eval: holdout (2026-08-24+) scored against REAL next-candle outcomes,
never touched by teacher or training.
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
LABELS = os.environ.get(
    "SML_DISTILL_LABELS",
    os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp/teacher_labels_qwen.jsonl"))
ADAPTER_OUT = os.environ.get(
    "SML_DISTILL_ADAPTER",
    os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp/adapter_distill"))
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"

TEST_START = pd.Timestamp("2026-08-24", tz="UTC")
MAX_TRAIN = 4000
EPOCHS = 2
BS = 8
GRAD_ACC = 4
LR = 1e-4
MAX_LEN = 96
TEMP = 2.0          # distillation temperature
ALPHA = 0.5         # CE vs KL mix


def load_labels():
    rows = []
    for line in open(LABELS):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("teacher_dir"):
            continue
        rows.append(r)
    # teacher labeled up to 2026-08-20; split ITS labels chronologically:
    # older 2/3 train (student learns teacher), newest 1/3 holdout (student
    # judged against REAL outcomes on states neither it nor the teacher used).
    rows.sort(key=lambda r: r["time"])
    cut = int(len(rows) * 2 / 3)
    return rows[:cut], rows[cut:]


def build_examples(rows):
    ex = []
    for r in rows:
        p_up = r["teacher_conf"] if r["teacher_dir"] == "UP" else 1 - r["teacher_conf"]
        ex.append({
            "prompt": r["state"] + " Next candle UP or DOWN? Answer with one word.",
            "hard": r["teacher_dir"],
            "p_up": p_up,
            "actual": r["actual_next"],
        })
    return ex


def main():
    train, hold = load_labels()
    print(f"teacher labels: train={len(train)} holdout={len(hold)}", flush=True)
    tr = build_examples(train)[:MAX_TRAIN]
    ho = build_examples(hold)

    # teacher's OWN accuracy on train vs its transfer potential
    teach_train = sum(1 for r in train if r["teacher_dir"] == r["actual_next"]) / max(1, len(train))
    teach_hold = sum(1 for r in hold if r["teacher_dir"] == r["actual_next"]) / max(1, len(hold))
    print(f"TEACHER accuracy: train {teach_train:.3f} | holdout {teach_hold:.3f}", flush=True)

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

    def btok(exs):
        texts = [f"{e['prompt']}\nAnswer: {e['hard']}" for e in exs]
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        labels = enc["input_ids"].clone()
        for i, e in enumerate(exs):
            pre = len(tok(e["prompt"] + "\nAnswer:")["input_ids"])
            labels[i, :pre - 1] = -100
        enc["labels"] = labels
        return {k: v.to(DEVICE) for k, v in enc.items()}

    # distillation loss: KL between model softmax over (UP|DOWN) and teacher's
    # distribution, computed on the final answer token position
    up_id = tok("UP")["input_ids"][0] if len(tok("UP")["input_ids"]) == 1 else None
    dn_id = tok("DOWN")["input_ids"][0] if len(tok("DOWN")["input_ids"]) == 1 else None
    up_token = tok.convert_ids_to_tokens(up_id) if up_id is not None else None
    dn_token = tok.convert_ids_to_tokens(dn_id) if dn_id is not None else None

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    idx = np.random.RandomState(7).permutation(len(tr))
    model.train()
    opt.zero_grad()
    steps = math.ceil(len(tr) / (BS * GRAD_ACC))
    running, nrun = 0.0, 0
    for bi in range(0, len(idx), BS):
        batch = [tr[j] for j in idx[bi:bi + BS]]
        enc = btok(batch)
        out = model(**enc)
        # CE on hard labels
        ce = out.loss / GRAD_ACC
        # KL on answer-token distribution (if token ids resolve)
        kl = torch.tensor(0.0, device=DEVICE)
        if up_id is not None and dn_id is not None:
            with torch.no_grad():
                logits = out.logits[:, -1, [up_id, dn_id]] / TEMP
                teacher_p = torch.tensor(
                    [[e["p_up"], 1 - e["p_up"]] for e in batch],
                    device=DEVICE)
                teacher_p = torch.clamp(teacher_p, 1e-6, 1 - 1e-6)
            log_p = torch.log_softmax(logits, dim=-1)
            kl = (teacher_p * (torch.log(teacher_p) - log_p)).sum(-1).mean()
        loss = (ALPHA * ce + (1 - ALPHA) * TEMP * TEMP * kl) / GRAD_ACC
        loss.backward()
        running += (ALPHA * ce.item() + (1 - ALPHA) * TEMP * TEMP * kl.item()) * GRAD_ACC
        nrun += 1
        if (bi // BS + 1) % GRAD_ACC == 0 or bi + BS >= len(idx):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
        if nrun % 200 == 0:
            print(f"  step {nrun}/{steps} loss {running/max(1,nrun):.4f}", flush=True)
    print(f"done loss {running/max(1,nrun):.4f}", flush=True)
    model.save_pretrained(ADAPTER_OUT)
    tok.save_pretrained(ADAPTER_OUT)
    print(f"student saved -> {ADAPTER_OUT}", flush=True)

    # HONEST EVAL on holdout vs real outcomes
    model.eval()
    correct = 0
    preds = []
    with torch.no_grad():
        for i in range(0, len(ho), 32):
            chunk = ho[i:i + 32]
            enc = tok([e["prompt"] + "\nAnswer:" for e in chunk],
                      truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(DEVICE)
            out = model.generate(**enc, max_new_tokens=1, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            toks = tok.batch_decode(out[:, -1].unsqueeze(1))
            for j, t in enumerate(toks):
                p = ("UP" if "UP" in t.upper() and "DOWN" not in t.upper() else "DOWN")
                preds.append(p)
                if p == chunk[j]["actual"]:
                    correct += 1
    acc = correct / max(1, len(ho))
    aup = sum(1 for e in ho if e["actual"] == "UP") / max(1, len(ho))
    print(f"\nHOLDOUT (real outcomes, n={len(ho)}): student {acc:.4f} | "
          f"always-up {aup:.4f} | coin 0.5", flush=True)
    print(f"VERDICT: {'PASS >55%' if acc > 0.55 else 'DEAD <=55%'}", flush=True)
    json.dump({"teacher_train": teach_train, "teacher_hold": teach_hold,
               "student_holdout_acc": acc, "always_up": aup,
               "holdout_n": len(ho)},
              open(os.path.join(HERE, "distill_results.json"), "w"))


if __name__ == "__main__":
    main()
