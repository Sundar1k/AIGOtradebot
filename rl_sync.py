import os
#!/usr/bin/env python3
"""rl_sync.py — Phase 5: 3-factor synchronized policy + 37% rule backtest.

Honest scope note: the REAL Chronos/XGBoost proba and the 7B veto cannot be
replayed over 92k candidates on this box (GPU model, 17s/call). So factor 2
and 3 are modeled from MEASURED relationships, clearly labeled:

  factor 1  rule direction (real, from the lane rules)
  factor 2  "proba band" proxy = the three measured selection conditions:
            |dist from level| in [0.5, 2.0] ATR  AND  ADX <= 40  AND
            ATR-vol not in the top tercile  (F6+F9+F10 — all beat
            baseline out-of-sample today)
  factor 3  veto agreement gate, stochastic with rate a (sensitivity
            50% / 62% measured / 75%)

37% rule (Optimal Stopping, applied as a principled heuristic):
  per evaluation window (100 candidates), the first 37% are calibration:
  record their synced score, take NO trades from them. Then EXPLOIT:
  take only candidates whose synced score STRICTLY beats the best
  calibration score. Sensitivity: 20% / 37% / 50%.

Out-of-sample split 2021-24 / 2025-26. Baseline A = sync w/o 37% rule.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import indicators as ind

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{symbol}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def candidates(df: pd.DataFrame):
    """All EMA+ORB signals with their synced-score factors (no filter)."""
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    t = df["time"]
    ef = ind.ema(c, config.EMA_FAST)
    es = ind.ema(c, config.EMA_SLOW)
    atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
    adx = np.asarray(ind.adx(df, config.ADX_P), dtype=float)
    oh, ol = ind.opening_range(df, config.ORB_BARS, config.ORB_OPEN_MIN, config.ORB_TZ)
    oh = np.asarray(oh, dtype=float)
    ol = np.asarray(ol, dtype=float)
    et_min = np.asarray(ind.et_minutes(df, config.ORB_TZ), dtype=float)
    n = len(c)
    atr_ok = atr[np.isfinite(atr) & (atr > 0)]
    p66 = np.percentile(atr_ok, 66) if len(atr_ok) else 1e9
    warm = max(80, config.ADX_P * 3)
    cands = []

    def settle(d, entry, risk, j0):
        stop = entry - d * risk
        tgt = entry + d * 2.0 * risk
        j = j0
        while j < n:
            if d > 0 and lo[j] <= stop:
                return -1.0, j
            if d < 0 and hi[j] >= stop:
                return -1.0, j
            if d > 0 and hi[j] >= tgt:
                return 2.0, j
            if d < 0 and lo[j] <= tgt:
                return 2.0, j
            j += 1
        return None, j

    def rec(strategy, side, r, i, dist, vol_ok):
        # synced score 0..3: one point per measured condition
        score = 0
        score += 1 if (0.5 <= abs(dist) <= 2.0) else 0
        score += 1 if adx[i] <= 40 else 0
        score += 1 if vol_ok else 0
        cands.append(dict(strategy=strategy, side=side, r=r, t=t.iloc[i],
                          score=score))

    i = warm
    while i < n - 1:
        if (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            if adx[i] >= config.ADX_GATE:
                if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
                    r, jx = settle(1, c[i], 0.5 * atr[i], i + 1)
                    if r is not None:
                        rec("ema", 1, r, i, (c[i] - es[i]) / atr[i], atr[i] <= p66)
                        i = jx + 1
                        continue
                elif ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
                    r, jx = settle(-1, c[i], 0.5 * atr[i], i + 1)
                    if r is not None:
                        rec("ema", -1, r, i, (c[i] - es[i]) / atr[i], atr[i] <= p66)
                        i = jx + 1
                        continue
        if (np.isfinite(oh[i]) and np.isfinite(oh[i - 1])
                and np.isfinite(ol[i]) and np.isfinite(ol[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            if et_min[i] < config.ORB_CLOSE_MIN and adx[i] >= config.ORB_ADX_GATE:
                if c[i - 1] <= oh[i - 1] and c[i] > oh[i]:
                    r, jx = settle(1, c[i], 0.5 * atr[i], i + 1)
                    if r is not None:
                        rec("orb", 1, r, i, (c[i] - oh[i]) / atr[i], atr[i] <= p66)
                        i = jx + 1
                        continue
                elif c[i - 1] >= ol[i - 1] and c[i] < ol[i]:
                    r, jx = settle(-1, c[i], 0.5 * atr[i], i + 1)
                    if r is not None:
                        rec("orb", -1, r, i, (c[i] - ol[i]) / atr[i], atr[i] <= p66)
                        i = jx + 1
                        continue
        i += 1
    return cands


def run_policy(cands, veto_rate, cutoff, window=100, seed=1):
    """Sync (+ optional 37% rule). Returns taken trades' R list."""
    rng = np.random.default_rng(seed)
    taken = []
    cal_max = -1            # best calibration score in current window
    in_cal = True
    count = 0
    for k, cd in enumerate(cands):
        if k % window == 0:
            cal_max, in_cal, count = -1, True, 0
        count += 1
        # factor 2: score >= 2 of 3 conditions = "in proba band" proxy
        if cd["score"] < 2:
            continue
        # factor 3: veto agreement gate
        if rng.random() > veto_rate:
            continue
        if in_cal:
            # 37% rule gathering phase: record best, take nothing
            cal_max = max(cal_max, cd["score"])
            if count >= max(1, int(window * cutoff)):
                in_cal = False
            continue
        # exploit: only strictly better than the calibration best
        if cd["score"] > cal_max:
            taken.append(cd["r"])
    return taken


def stats(rs):
    if not rs:
        return dict(n=0, wr=0.0, avg=0.0, sm=0.0, pf=0.0)
    rs = np.array(rs)
    n = len(rs)
    wr = float(np.sum(rs > 0)) / n
    avg = float(np.mean(rs))
    g_win = float(np.sum(rs[rs > 0]))
    g_loss = -float(np.sum(rs[rs < 0]))
    pf = g_win / g_loss if g_loss > 0 else float("inf")
    return dict(n=n, wr=wr, avg=avg, sm=float(np.sum(rs)), pf=pf)


def main():
    print("=== PHASE 5 — 3-FACTOR SYNC + 37% RULE (out-of-sample 2025-26) ===", flush=True)
    print("factor2 proxy = measured F6+F9+F10 conditions · factor3 = veto rate gate", flush=True)
    print("=" * 96, flush=True)
    all_c = []
    for s in SYMBOLS:
        all_c += candidates(load(s))
    oos = [cd for cd in all_c if cd["t"] >= OOS_START]
    print(f"candidates total {len(all_c)} · out-of-sample {len(oos)}", flush=True)

    print(f"\n{'config':<34} {'n':>6} {'WR':>7} {'avgR':>8} {'sumR':>8} {'PF':>6}", flush=True)
    print("-" * 96, flush=True)
    base = run_policy(oos, veto_rate=0.62, cutoff=0.0)   # sync, no 37%
    st = stats(base)
    print(f"{'A: sync only (no 37%)':<34} {st['n']:>6} {st['wr']:>7.1%} "
          f"{st['avg']:>+8.3f} {st['sm']:>+8.0f} {st['pf']:>6.2f}", flush=True)
    for rate in (0.50, 0.62, 0.75):
        for cutoff in (0.20, 0.37, 0.50):
            taken = run_policy(oos, veto_rate=rate, cutoff=cutoff)
            st = stats(taken)
            beats = (st["n"] >= 100 and st["avg"] > stats(base)["avg"]
                     and st["pf"] > stats(base)["pf"])
            tag = "✓" if beats else " "
            print(f"{'B: veto '+str(rate)+' cutoff '+str(cutoff):<34} {st['n']:>6} "
                  f"{st['wr']:>7.1%} {st['avg']:>+8.3f} {st['sm']:>+8.0f} "
                  f"{st['pf']:>6.2f} {tag}", flush=True)
    print("=" * 96, flush=True)
    print("Verdict rule: n>=100 AND avgR > sync-only AND PF > sync-only.", flush=True)
    print("(Veto rate 0.62 = measured 7B agreement; 0.50/0.75 = sensitivity.)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
