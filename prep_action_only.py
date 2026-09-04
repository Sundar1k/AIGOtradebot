#!/usr/bin/env python3
"""prep_action_only.py — build the ACTION-ONLY retrain dataset for the 1.5B.

The failed 1.5B student (11.4% agreement) produced rationale-dominated
completions where the action token got lost. This strips each teacher
output to JUST the action (BUY/SELL/NO TRADE) and rebuilds the alpaca
JSONL, so the retrained student has nothing to generate but the action.

Reads:  finetune/distill_train_alpaca.jsonl, finetune/distill_val_alpaca.jsonl
Writes: finetune/action_train_alpaca.jsonl, finetune/action_val_alpaca.jsonl
        (train output: "SELL 1 contract" -> "SELL" ; val identical)

Run: ./.venv/bin/python prep_action_only.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [("distill_train_alpaca.jsonl", "action_train_alpaca.jsonl"),
       ("distill_val_alpaca.jsonl", "action_val_alpaca.jsonl")]

ACTION_RE = re.compile(r"^\s*(BUY|SELL|NO TRADE|NO)", re.I)


def action_of(out: str) -> str:
    m = ACTION_RE.match(out or "")
    if not m:
        return "NO TRADE"          # unparseable -> neutral (safe, fail-closed)
    w = m.group(1).upper()
    if w == "NO":
        return "NO TRADE"
    return w


def main():
    for src, dst in SRC:
        srcp = os.path.join(HERE, "finetune", src)
        dstp = os.path.join(HERE, "finetune", dst)
        if not os.path.exists(srcp):
            print(f"skip {src} (missing)")
            continue
        n = 0
        dist = {}
        with open(srcp) as f, open(dstp, "w") as g:
            for line in f:
                d = json.loads(line)
                act = action_of(d["output"])
                dist[act] = dist.get(act, 0) + 1
                g.write(json.dumps({"instruction": d["instruction"],
                                    "output": act}) + "\n")
                n += 1
        print(f"{src} -> {dst}: {n} rows, dist={dist}")
    print("DONE — ready for soup15b-action.yaml retrain (GPU, weekend window)")


if __name__ == "__main__":
    main()
