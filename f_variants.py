import os
#!/usr/bin/env python3
"""f_variants.py — Phase 4 flow-experiment variants vs baseline (out-of-sample).

Variants tested (from today's conditional search):
  F6  VOL GATE        — skip entries in high ATR-percentile regimes
  F9  EMA PLACEMENT   — skip EMA entries |dist from EMA20| <0.5 or >2.0 ATR
  F10 ADX CEILING     — skip entries when ADX > 40
  F11 ORB EXTENSION   — ONLY take ORB breakouts with |ext| in [1.3, 1.8] ATR

Pre-registered accept rule: out-of-sample (2025-26) must beat baseline in
BOTH avgR and PF, with n >= 100. Same data, same costs, no look-ahead.
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


def simulate(df: pd.DataFrame, variant: str):
    """Same rules as ema_orb_conditional.simulate + optional variant filter.
    Returns list of (strategy, side, r, time)."""
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
    trades = []

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

    def ema_pass(i, d):
        if variant == "F9_EMAPLACE":
            dist = abs((c[i] - es[i]) / atr[i]) if atr[i] > 0 else 0
            return 0.5 <= dist <= 2.0
        if variant == "F10_ADXCEIL":
            return adx[i] <= 40
        if variant == "F6_VOLGATE":
            return atr[i] <= p66
        return True

    def orb_pass(i, d):
        if variant == "F11_ORBEXT":
            level = oh[i] if d > 0 else ol[i]
            dist = abs((c[i] - level) / atr[i]) if atr[i] > 0 else 0
            return 1.3 <= dist <= 1.8
        if variant == "F6_VOLGATE":
            return atr[i] <= p66
        return True

    i = warm
    while i < n - 1:
        if (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            if adx[i] >= config.ADX_GATE:
                if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
                    if ema_pass(i, 1):
                        r, jx = settle(1, c[i], 0.5 * atr[i], i + 1)
                        if r is not None:
                            trades.append(("ema", 1, r, t.iloc[i]))
                            i = jx + 1
                            continue
                elif ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
                    if ema_pass(i, -1):
                        r, jx = settle(-1, c[i], 0.5 * atr[i], i + 1)
                        if r is not None:
                            trades.append(("ema", -1, r, t.iloc[i]))
                            i = jx + 1
                            continue
        if (np.isfinite(oh[i]) and np.isfinite(oh[i - 1])
                and np.isfinite(ol[i]) and np.isfinite(ol[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            if et_min[i] < config.ORB_CLOSE_MIN and adx[i] >= config.ORB_ADX_GATE:
                if c[i - 1] <= oh[i - 1] and c[i] > oh[i]:
                    if orb_pass(i, 1):
                        r, jx = settle(1, c[i], 0.5 * atr[i], i + 1)
                        if r is not None:
                            trades.append(("orb", 1, r, t.iloc[i]))
                            i = jx + 1
                            continue
                elif c[i - 1] >= ol[i - 1] and c[i] < ol[i]:
                    if orb_pass(i, -1):
                        r, jx = settle(-1, c[i], 0.5 * atr[i], i + 1)
                        if r is not None:
                            trades.append(("orb", -1, r, t.iloc[i]))
                            i = jx + 1
                            continue
        i += 1
    return trades


def stats(rows):
    if not rows:
        return dict(n=0, wr=0.0, avg=0.0, sm=0.0, pf=0.0)
    rs = np.array([r[2] for r in rows])
    n = len(rs)
    wr = float(np.sum(rs > 0)) / n
    avg = float(np.mean(rs))
    g_win = float(np.sum(rs[rs > 0]))
    g_loss = -float(np.sum(rs[rs < 0]))
    pf = g_win / g_loss if g_loss > 0 else float("inf")
    return dict(n=n, wr=wr, avg=avg, sm=float(np.sum(rs)), pf=pf)


def main():
    print("=== PHASE 4 VARIANTS — out-of-sample 2025-26 vs baseline ===", flush=True)
    print("rule: EMA9/20 + ADX>=18 · ORB 15min · stop 0.5xATR · target 2R", flush=True)
    print("=" * 96, flush=True)
    variants = ["BASE", "F6_VOLGATE", "F9_EMAPLACE", "F10_ADXCEIL", "F11_ORBEXT"]
    results = {v: [] for v in variants}
    for s in SYMBOLS:
        df = load(s)
        for v in variants:
            trades = simulate(df, None if v == "BASE" else v)
            oos = [t for t in trades if t[3] >= OOS_START]
            results[v].extend(oos)
    base = stats(results["BASE"])
    print(f"\n{'variant':<14} {'n':>7} {'WR':>7} {'avgR':>8} {'sumR':>9} {'PF':>6}  verdict", flush=True)
    print("-" * 96, flush=True)
    for v in variants:
        st = stats(results[v])
        if v == "BASE":
            verdict = "baseline"
        else:
            beats = (st["n"] >= 100 and st["avg"] > base["avg"] and st["pf"] > base["pf"])
            verdict = "✓ BEATS BASELINE" if beats else "✗ does not beat"
        print(f"{v:<14} {st['n']:>7} {st['wr']:>7.1%} {st['avg']:>+8.3f} {st['sm']:>+9.0f} "
              f"{st['pf']:>6.2f}  {verdict}", flush=True)
    print("=" * 96, flush=True)
    print("Accept rule: n>=100 AND avgR > baseline AND PF > baseline (out-of-sample).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
