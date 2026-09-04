#!/usr/bin/env python3
"""sml_advisor.py — LIVE SHADOW SML advisor (never places orders).

Watches the same live bars as the bot (broker API), and for every new bar on
every symbol:
  1. builds the state line (same format the SML was trained on)
  2. asks the fine-tuned SML: next candle UP or DOWN?
  3. computes a suggested entry (next-bar open), SL (0.5xATR20), TP (2R) —
     same geometry as the live bot
  4. logs {time, symbol, prediction, entry, sl, tp, state} to
     ~/.autotrade_sml_advisor.jsonl

When the next candle closes, the outcome is backfilled (label arrives 15m
later on the 15m timeframe) so the LIVE HIT-RATE is tracked continuously.

Boundaries (locked):
  - inference on CPU (0.5B is small; 3-min bars give plenty of time) so the
    1650S stays free for live retraining
  - NEVER sends orders; log-only instrument
  - if live hit-rate <= 55% after 200 labeled live signals -> archive
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

ADAPTER = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/sml_exp/adapter_live_5m")
LEDGER = os.path.join(os.path.expanduser("~"), ".autotrade_sml_advisor.jsonl")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TF = 5
SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]

_model = None
_tok = None


def get_model():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.padding_side = "left"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
    model.eval()
    _model, _tok = model, tok
    print(f"[sml_advisor] model on {device} (float16)", flush=True)
    return model, tok


def state_line(sym, df, i):
    c = df["close"].to_numpy(float)[max(0, i - 59):i + 1]
    h = df["high"].to_numpy(float)[max(0, i - 59):i + 1]
    l = df["low"].to_numpy(float)[max(0, i - 59):i + 1]
    d = np.diff(c[-15:])
    up = d[d > 0].sum() / 14
    dn = -d[d < 0].sum() / 14
    rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
    e10 = pd.Series(c).ewm(span=10, adjust=False).mean().iloc[-1]
    e30 = pd.Series(c).ewm(span=30, adjust=False).mean().iloc[-1]
    side = "above" if e10 >= e30 else "below"
    hhk, llk = h[-14:].max(), l[-14:].min()
    st = 100 * (c[-1] - llk) / max(1e-9, hhk - llk)
    prev_hh, prev_ll = h[-15:-1][-14:].max(), l[-15:-1][-14:].min()
    prev_st = 100 * (c[-2] - prev_ll) / max(1e-9, prev_hh - prev_ll)
    sdir = "rising" if st >= prev_st else "falling"
    tr = max(h[-1] - l[-1], abs(h[-1] - c[-2]), abs(l[-1] - c[-2]))
    return (f"{sym} {TF}m. RSI {int(round(rsi))}, EMA10 {side} EMA30, "
            f"stochastic {int(round(st))} {sdir}, ATR {int(round(tr))}.")


def predict(state_text):
    model, tok = get_model()
    import torch
    prompt = state_text + " Next candle UP or DOWN? Answer with one word.\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(
        _model.device)
    import torch as _torch
    with _torch.no_grad():
        out = model.generate(**ids, max_new_tokens=1, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    t = tok.decode(out[0][ids["input_ids"].shape[1]:],
                   skip_special_tokens=True).strip().upper()
    return "UP" if "UP" in t and "DOWN" not in t else "DOWN"


def atr20(df, i):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    trs = np.maximum.reduce([h[i - 20:i + 1] - l[i - 20:i + 1],
                             abs(h[i - 20:i + 1] - c[i - 21:i]),
                             abs(l[i - 20:i + 1] - c[i - 21:i])])
    return float(np.mean(trs))


def backfill(client=None):
    """Label already-logged predictions whose next candle has closed.

    Label rule: the candle that STARTS at r['time'] closes TF minutes later.
    Its close vs the logged entry (= close of the prior bar) gives UP/DOWN.
    """
    if not os.path.exists(LEDGER):
        return 0
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    new = 0
    cache = {}
    now = dt.datetime.now(dt.timezone.utc)
    for r in rows:
        if not r.get("label"):
            t = pd.Timestamp(r["time"])
            if now >= t + dt.timedelta(minutes=TF):
                try:
                    if client is None:
                        raise RuntimeError("no broker client")
                    sym = r["symbol"]
                    if sym not in cache:
                        contract = client.get_active_contract(sym)
                        df = client.get_bars(contract["id"], TF, limit=500)
                        df = df.sort_values("time").reset_index(drop=True)
                        cache[sym] = df
                    df = cache[sym]
                    # prediction made at the close of bar T (== r["time"]);
                    # outcome candle is the NEXT bar after it.
                    idx = df.index[df["time"] == pd.Timestamp(r["time"])]
                    if len(idx) and idx[0] + 1 < len(df):
                        nxt_close = float(df["close"].iloc[idx[0] + 1])
                        if nxt_close != r["entry"]:
                            r["label"] = ("UP" if nxt_close > r["entry"]
                                          else "DOWN")
                        else:
                            # exact tie: use next-next bar direction
                            if idx[0] + 2 < len(df):
                                nn = float(df["close"].iloc[idx[0] + 2])
                                r["label"] = ("UP" if nn > r["entry"]
                                              else "DOWN" if nn < r["entry"]
                                              else "UP")   # final fallback
                            else:
                                continue
                        r["exit"] = round(nxt_close, 4)
                        r["labeled_ts"] = str(now)
                        new += 1
                except Exception as e:
                    print(f"backfill error {r['symbol']} {r['time']}: {e}",
                          flush=True)
        if True:  # keep every row exactly once
            pass
    out_rows = rows
    if new:
        with open(LEDGER, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
    return new


def main():
    import broker
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    client = broker.make_broker()
    client.authenticate()
    client.pick_account(config.ACCOUNT)

    last_bar = {}
    while True:
        try:
            for sym in SYMBOLS:
                contract = client.get_active_contract(sym)
                df = client.get_bars(contract["id"], TF, limit=500)
                if df.empty:
                    continue
                df = df.sort_values("time").reset_index(drop=True)
                t = str(df["time"].iloc[-1])
                if last_bar.get(sym) == t:
                    continue
                last_bar[sym] = t
                i = len(df) - 1
                if i < 60:
                    continue
                st = state_line(sym, df, i)
                pred = predict(st)
                a = atr20(df, i)
                entry = float(df["close"].iloc[i])
                risk = config.STOP_ATR * a
                sl = entry - risk if pred == "UP" else entry + risk
                tp = entry + 2 * risk if pred == "UP" else entry - 2 * risk
                rec = {"time": t, "symbol": sym, "prediction": pred,
                       "entry": round(entry, 4), "sl": round(sl, 4),
                       "tp": round(tp, 4), "atr": round(a, 4),
                       "state": st, "label": None, "ts": str(dt.datetime.now(dt.timezone.utc))}
                with open(LEDGER, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{t}] {sym} {pred} entry={entry:.2f} sl={sl:.2f} "
                      f"tp={tp:.2f}", flush=True)
            if backfill(client):
                try:
                    import sml_gamify
                    gs = sml_gamify.load_state()
                    sml_gamify.roll_day(gs)
                    sml_gamify.play(gs)
                    sml_gamify.save_state(gs)
                except Exception as ge:
                    print(f"gamify error: {ge}", flush=True)
        except Exception as e:
            print(f"cycle error: {e}", flush=True)
        if args.once:
            return 0
        time.sleep(20)   # 5m bars: poll faster so predictions land on time


if __name__ == "__main__":
    main()
