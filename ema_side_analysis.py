import os
#!/usr/bin/env python3
"""ema_side_analysis.py — EMA long vs short + entry-placement analysis.

Answers: for EMA9/20 cross signals (same rule as the live bot, ADX>=18,
stop 0.5xATR(20), target 2R):
  1. LONG vs SHORT: which side wins more?
  2. Entry PLACEMENT: how far (in ATR units) from EMA20 was the entry,
     and does chasing extended entries lose?
Full history, all 5 symbols, no look-ahead.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import indicators as ind

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{symbol}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def run(df: pd.DataFrame):
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    ef = ind.ema(c, config.EMA_FAST)
    es = ind.ema(c, config.EMA_SLOW)
    atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
    adx = np.asarray(ind.adx(df, config.ADX_P), dtype=float)
    n = len(c)
    warm = max(80, config.ADX_P * 3)
    trades = []
    i = warm
    while i < n - 1:
        if not (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            i += 1
            continue
        if config.ADX_GATE and adx[i] < config.ADX_GATE:
            i += 1
            continue
        if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
            d = 1
        elif ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
            d = -1
        else:
            i += 1
            continue
        risk = 0.5 * atr[i]
        entry = c[i]
        stop = entry - d * risk
        tgt = entry + d * 2.0 * risk
        r = None
        j_exit = -1
        j = i + 1
        while j < n:
            if d > 0 and lo[j] <= stop:
                r, j_exit = -1.0, j
                break
            if d < 0 and hi[j] >= stop:
                r, j_exit = -1.0, j
                break
            if d > 0 and hi[j] >= tgt:
                r, j_exit = 2.0, j
                break
            if d < 0 and lo[j] <= tgt:
                r, j_exit = 2.0, j
                break
            j += 1
        if r is None or j_exit < 0:
            break
        ema_dist = (entry - es[i]) / atr[i]        # + = price above EMA20
        trades.append((d, ema_dist, r))
        i = j_exit + 1
    return trades


def summarize(rows, label):
    if not rows:
        print(f"  {label:<28} no trades")
        return
    rs = [r[2] for r in rows]
    n = len(rs)
    wr = sum(1 for x in rs if x > 0) / n
    avg = sum(rs) / n
    g_win = sum(x for x in rs if x > 0)
    g_loss = -sum(x for x in rs if x < 0)
    pf = g_win / g_loss if g_loss > 0 else float("inf")
    print(f"  {label:<28} n={n:>6}  WR {wr:>6.1%}  avg {avg:+.3f}R  sum {sum(rs):+.0f}R  PF {pf:5.2f}")


def main():
    print("=== EMA 9/20 CROSS — LONG vs SHORT + ENTRY PLACEMENT (5y, all symbols) ===", flush=True)
    print("rule: ADX>=18 · stop 0.5xATR(20) · target 2R · no ML, no veto, no look-ahead", flush=True)
    print("=" * 100, flush=True)

    all_trades = []
    for s in SYMBOLS:
        all_trades += run(load(s))

    longs = [t for t in all_trades if t[0] > 0]
    shorts = [t for t in all_trades if t[0] < 0]

    print("\n--- 1. WHICH SIDE WINS? ---", flush=True)
    summarize(longs, "LONG (EMA cross up)")
    summarize(shorts, "SHORT (EMA cross down)")
    summarize(all_trades, "ALL")

    print("\n--- 2. ENTRY PLACEMENT (|entry − EMA20| in ATR units) ---", flush=True)
    for lo_, hi_, lab in [(0, 0.25, "touching EMA (<0.25 ATR)"),
                          (0.25, 0.5, "close (0.25-0.5 ATR)"),
                          (0.5, 1.0, "mid (0.5-1.0 ATR)"),
                          (1.0, 2.0, "extended (1-2 ATR)"),
                          (2.0, 99.0, "chasing (2+ ATR)")]:
        rows = [t for t in all_trades if lo_ <= abs(t[1]) < hi_]
        summarize(rows, f"|dist| {lab}")

    print("\n--- 3. PLACEMENT x SIDE (the money question) ---", flush=True)
    for side, sl in (("LONG", longs), ("SHORT", shorts)):
        for lo_, hi_, lab in [(0, 0.5, "near EMA"),
                              (0.5, 1.5, "mid"),
                              (1.5, 99.0, "extended")]:
            rows = [t for t in sl if lo_ <= abs(t[1]) < hi_]
            summarize(rows, f"{side} {lab}")
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
