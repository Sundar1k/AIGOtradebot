#!/usr/bin/env python3
"""fast_veto.py — the distilled XGBoost veto (reject-only pre-filter).

The 7B veto is distilled into a 3µs XGBoost student (finetune/veto_student.json,
trained by finetune/train_live_student.py on every teacher-labeled state we
have). This module answers the SAME question the 7B answers — BUY/SELL/NO TRADE
— from the v1 state line's numeric features.

SAFETY CONTRACT (the only way it's allowed in the live path):
  * REJECT-ONLY. It can only say "skip the GPU and block", never "approve".
    It fires ONLY when P(NO TRADE) >= AUTOTRADE_FAST_VETO_THRESH (default 0.95)
    — measured 97.3% precision on holdout (i.e. when it rejects, the 7B would
    have rejected 97.3% of the time). The 2.7% miss is a missed trade, never a
    bad entry.
  * FAIL-OPEN to the 7B. Any error → confident_reject=False → the caller falls
    through to the real veto. It can degrade to a no-op, never to a wrong gate.
  * The 7B remains the authority for every APPROVAL. This module never approves.

Loaded lazily + reloaded when the model file's mtime changes (so a cron retrain
swaps in on the next signal without a supervisor restart).
"""
import os
import re

import numpy as np
import xgboost as xgb

MODEL = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/finetune/veto_student.json")
SYMS = ["NQ", "ES", "RTY", "YM", "GC"]
ACTIONS = ("BUY", "SELL", "NO TRADE")
THRESHOLD = float(os.environ.get("AUTOTRADE_FAST_VETO_THRESH", "0.95"))

_inst = None
_inst_mtime = None


def _parse(state_text):
    """'ES 3m. RSI 55, EMA10 above EMA30, stochastic 55 rising, ATR 2. Score +1.'
    -> 1x12 feature array (same 12 features train_live_student.py learns on)."""
    sym = (state_text or "").split()[0]
    feats = [1.0 if s == sym else 0.0 for s in SYMS]
    m = re.search(r"RSI (\d+)", state_text)
    feats.append(float(m.group(1)) if m else np.nan)
    feats.append(1.0 if "EMA10 above" in state_text else 0.0)
    m = re.search(r"stochastic (\d+)", state_text)
    feats.append(float(m.group(1)) if m else np.nan)
    feats.append(1.0 if " rising" in state_text else 0.0)
    m = re.search(r"ATR (\d+)", state_text)
    feats.append(float(m.group(1)) if m else np.nan)
    m = re.search(r"Score ([+-]?\d+)", state_text)
    feats.append(float(m.group(1)) if m else np.nan)
    return np.array([feats], dtype=float)


def _load():
    global _inst, _inst_mtime
    mtime = os.path.getmtime(MODEL)
    if _inst is not None and mtime == _inst_mtime:
        return _inst
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL)
    _inst, _inst_mtime = clf, mtime
    return _inst


def decide_fast(state_text):
    """Return dict(action, p_no_trade, confident_reject). Never raises — on any
    error returns confident_reject=False so the caller falls through to the 7B."""
    try:
        clf = _load()
        p = clf.predict_proba(_parse(state_text))[0]   # [BUY, SELL, NO TRADE]
        p_no = float(p[2])
        return {"action": ACTIONS[int(p.argmax())],
                "p_no_trade": p_no,
                "confident_reject": p_no >= THRESHOLD}
    except Exception as e:
        return {"action": "NO TRADE", "p_no_trade": 0.0,
                "confident_reject": False, "error": str(e)}


if __name__ == "__main__":
    for t in (
        "ES 3m. RSI 45, EMA10 above EMA30, stochastic 62 rising, ATR 8. Score +2.",
        "NQ 3m. RSI 68, EMA10 above EMA30, stochastic 95 falling, ATR 6. Score +1.",
        "GC 3m. RSI 32, EMA10 below EMA30, stochastic 20 rising, ATR 4. Score -3.",
    ):
        print(f"{t}\n  -> {decide_fast(t)}\n")
