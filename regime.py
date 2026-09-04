#!/usr/bin/env python3
"""regime.py — market regime detector (2-3 state HMM) for the autotrader.

Implements the Adaptive-Markets remedy (Lo 2004) in its modern ML form:
fit an online Hidden Markov Model on daily returns (vol + trend states),
classify the CURRENT regime, and let the supervisor gate entries:

  * REGIME_GATE=strict (default): trade only when the current regime matches
    the regimes the veto model was trained in (calm/trending). In a
    "crisis"/high-vol regime the bot idles — the patterns it learned don't
    exist there.
  * REGIME_GATE=off: disable (trades in any regime, current behavior).

Design:
  - Features: daily close-to-close log returns + realized vol (rolling std),
    the two things that separate calm/trend/panic regimes.
  - 3-state GaussianHMM (calm, trending, high-vol), retrained on a rolling
    window (2y) each time it runs, so it adapts to drift (AMH).
  - Output: ~/.autotrade_regime.json {regime, prob, gate, ts} —
    read by supervisor.py each bar (cheap), written on each run.
  - Runs standalone: `regime.py --update` (cron every 6h) or in-process.

Fit cost on GTX 1070 / i7: ~seconds for 2y of daily bars.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(os.path.expanduser("~"), ".autotrade_regime.json")
WINDOW_DAYS = 730                # 2y rolling fit window
MIN_BARS = 400                   # need at least this many daily closes

# Which HMM state maps to which trading regime (state index -> label).
# Labels are assigned AFTER fit by ordering states on mean return + vol:
#   0 = calm (low vol, drift ~0)
#   1 = trending (directional, moderate vol)
#   2 = panic / high-vol (elevated vol, negative drift)


def load_daily_returns(symbol: str) -> pd.Series:
    """Daily log returns from the bot's 3-min CSV (resampled)."""
    path = f"{BASE}/data/{symbol}_3min.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    daily = df.set_index("datetime")["close"].resample("1D").last().dropna()
    rets = np.log(daily).diff().dropna()
    return rets


def fit_hmm(rets: pd.Series, n_states: int = 3, seed: int = 42) -> tuple:
    """Fit a GaussianHMM on (return, realized-vol) features. Returns
    (model, labels_dict) where labels_dict maps state_idx -> regime name."""
    from hmmlearn import hmm

    vol = rets.rolling(20).std().dropna()
    feat = pd.DataFrame({"r": rets, "v": vol}).dropna()
    X = feat.to_numpy(float)

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=200, random_state=seed, tol=1e-4)
    model.fit(X)

    # Assign regime labels by (mean return, mean vol) of each state
    states = []
    for k in range(n_states):
        m = model.means_[k]
        states.append({"idx": k, "mean_r": float(m[0]), "mean_v": float(m[1])})
    # order: lowest vol = calm ... highest vol = panic
    states.sort(key=lambda s: s["mean_v"])
    names = ["calm", "trending", "panic"]
    for i, s in enumerate(states):
        s["regime"] = names[i] if i < len(names) else f"state{i}"
    labels = {s["idx"]: s["regime"] for s in states}
    return model, labels


def current_regime(symbol: str = "NQ") -> dict:
    """Fit (or reuse cache) and classify the most recent day. Returns the
    state dict for supervisor: {regime, prob, gate, ts, vol}."""
    rets = load_daily_returns(symbol)
    if len(rets) < MIN_BARS:
        return {"regime": "unknown", "prob": 0.0, "gate": "off",
                "reason": f"only {len(rets)} daily bars", "ts": _now()}

    rets = rets.tail(WINDOW_DAYS)
    vol = rets.rolling(20).std().dropna()
    feat = pd.DataFrame({"r": rets, "v": vol}).dropna()
    X = feat.to_numpy(float)

    model, labels = fit_hmm(rets)
    probs = model.predict_proba(X)
    last = probs[-1]
    state = int(np.argmax(last))
    regime = labels[state]
    prob = float(last[state])

    # trailing vol for context (annualized %)
    ann_vol = float(vol.iloc[-1] * np.sqrt(252) * 100) if len(vol) else 0.0
    return {"regime": regime, "prob": round(prob, 3), "gate": "on",
            "symbol": symbol, "ann_vol_pct": round(ann_vol, 1),
            "ts": _now(), "state_means": labels}


def _now() -> str:
    return dt.datetime.now().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--update", action="store_true",
                    help="fit + classify + write state file (cron)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    st = current_regime(args.symbol)
    if args.verbose or not args.update:
        print(json.dumps(st, indent=2))

    if args.update:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, STATE_FILE)
        print(f"regime state written: {STATE_FILE} -> {st['regime']} "
              f"(prob {st['prob']})", flush=True)


if __name__ == "__main__":
    main()
