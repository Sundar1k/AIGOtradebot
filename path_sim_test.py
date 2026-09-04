#!/usr/bin/env python3
"""path_sim_test.py — pre-registered test: does the Chronos forecast band
predict which ledger signals win their 2R/1R bracket?

Pre-registered gates (2026-08-25, BEFORE running — locked):
  G1 rank corr (analysis set):      Spearman(p_win, r) > 0.15
  G2 holdout separation:            top-half WR >= bottom-half WR + 8pts
                                    AND top meanR - bottom meanR >= +0.25R
  G3 holdout gate at analysis-set best threshold:
                                    p(P(meanR<=0)) < 0.20 (bootstrap 10k, seed 42),
                                    n_kept >= 30
  G4 sanity:                        corr(p_win, proba) < 0.7 and
                                    corr(p_win, r_hat) < 0.7

Kill criteria: any fail -> DEAD, logged, no re-runs with tweaks.

Method note: chronos-bolt is QUANTILE-based (no native path sampling). We use
its 9 trained quantile levels [0.1..0.9] over the hold horizon as a discrete
distribution of outcomes — the honest equivalent of "fraction of paths that
hit +2R before -1R". Bracket geometry matches missed_trades.simulate exactly:
entry = signal bar close, stop = entry - dir*STOP_ATR*ATR20, target =
entry + dir*RR*risk, stop wins ties, MAX_HOLD_BARS=60 then close-out.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(os.path.expanduser("~"), ".autotrade_missed.json")
OUT = os.path.join(os.path.expanduser("~"), ".autotrade_pathsim.json")
MAX_HOLD = int(os.environ.get("AUTOTRADE_LEARN_HOLD", "60"))
QUANT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_bars_upto(symbol: str) -> pd.DataFrame:
    """CSV history + fresh API bars merged (same as missed_trades.load_bars)."""
    import datetime as dt
    from missed_trades import load_bars
    df = load_bars(symbol)
    return df.sort_values("time").reset_index(drop=True)


def bracket_from_paths(qpaths: np.ndarray, entry: float, stop: float,
                       target: float, direction: int) -> float:
    """Score quantile 'paths' like the real bracket.

    qpaths: [H, Q] forecast prices for each future bar at each quantile level.
    Each quantile column acts as one path; first touch wins (stop wins ties,
    matching sim_broker/missed_trades conservative rule). Returns win fraction.
    """
    hits = np.where(qpaths <= stop if direction > 0 else qpaths >= stop)
    tg = np.where(qpaths >= target if direction > 0 else qpaths <= target)
    wins = losses = 0
    for q in range(qpaths.shape[1]):
        h_bar = hits[0][hits[1] == q].min() if (hits[1] == q).any() else 10**9
        t_bar = tg[0][tg[1] == q].min() if (tg[1] == q).any() else 10**9
        if t_bar < h_bar:
            wins += 1
        else:
            losses += 1          # stop-first or timeout -> conservative loss
    return wins / max(1, wins + losses)


def main():
    t0 = time.time()
    d = json.load(open(LEDGER))
    recs = d["records"] if isinstance(d, dict) else d
    print(f"ledger: {len(recs)} records", flush=True)

    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny", device_map="cpu", dtype=torch.float32)

    bars_cache = {}
    results = []
    for n, r in enumerate(recs):
        sym = r["symbol"]
        if sym not in bars_cache:
            bars_cache[sym] = load_bars_upto(sym)
        df = bars_cache[sym]

        tsig = pd.Timestamp(r["time"])
        i = df.index[df["time"] == tsig]
        if len(i) == 0:
            # find nearest bar within tolerance (CSV vs API timestamp drift)
            j = (df["time"] - tsig).abs().idxmin()
            if abs((df["time"].iloc[j] - tsig).total_seconds()) > 180:
                continue
            i = [j]
        i = int(i[0])
        if i < config.CTX or i + 2 >= len(df):
            continue

        ctx = np.log(df["close"].to_numpy(float)[i - config.CTX + 1:i + 1])
        x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        # ATR20 at the signal bar (same as live stop geometry)
        h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
        tr = np.maximum(h[i] - l[i],
             np.maximum(abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        atr = pd.Series(np.maximum.reduce([
            h[max(0, i - config.ATR_P):i + 1] - l[max(0, i - config.ATR_P):i + 1],
            abs(h[max(0, i - config.ATR_P):i + 1] - c[max(0, i - config.ATR_P):i + 1]),
            abs(l[max(0, i - config.ATR_P):i + 1] - c[max(0, i - config.ATR_P):i + 1]),
        ])).mean()
        entry = c[i]
        direction = int(r["dir"])
        risk = float(config.STOP_ATR * atr)
        if risk <= 0:
            continue
        stop = entry - direction * risk
        target = entry + direction * risk * config.RR

        with torch.no_grad():
            q, _mean = pipe.predict_quantiles(
                inputs=x, prediction_length=MAX_HOLD,
                quantile_levels=QUANT_LEVELS)
        qp = q[0].numpy()                     # [H, 9]
        # de-normalize: bolt forecasts are in normalized space? No — predict_quantiles
        # returns values in INPUT scale already.
        p_win = bracket_from_paths(qp, entry, stop, target, direction)
        results.append({"symbol": sym, "time": str(tsig), "dir": direction,
                        "proba": r["proba"], "r_hat": r["r_hat"],
                        "r": r["r"], "kind": r.get("kind"),
                        "atr": round(float(atr), 4), "risk": round(risk, 4),
                        "p_win": round(float(p_win), 4)})
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(recs)} scored ({time.time()-t0:.0f}s)", flush=True)

    json.dump(results, open(OUT, "w"))
    print(f"\nwrote {len(results)} scored signals -> {OUT} "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
