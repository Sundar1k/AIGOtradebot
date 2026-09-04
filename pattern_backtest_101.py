import os
#!/usr/bin/env python3
"""pattern_backtest_101.py — full candlestick-pattern -> next-candle -1/0/1 backtest.

Answers the user's exact question: "after backtesting candlestick patterns in
-1/0/1, does the SLM have enough data to see the pattern and predict?"

For every 30-min candle across all 5 symbols (5yr history):
  detect patterns (single + two-bar, from candle_patterns.py)
  record the NEXT 30-min candle's direction:
      +1 = closed HIGHER,  -1 = closed LOWER,  0 = same level (tie, near-never)

Then per pattern: sample count + P(next up / down) and does the pattern's
implied direction actually come true more than the ~50% baseline?
No look-ahead: the next candle is strictly AFTER the pattern candle closes.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candle_patterns import resample_30m, detect_on_candle, PATTERN_DIR

SYM = ["NQ", "ES", "RTY", "YM", "GC"]
EPS = 0.0001  # "same level" tie band (1 bp) — 30m exact ties are near-never


def load(sym):
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def label(diff):
    if diff > 0:
        return 1
    if diff < 0:
        return -1
    return 0


def main():
    print("=== CANDLESTICK PATTERN -> NEXT 30m CANDLE (-1/0/1) BACKTEST ===", flush=True)
    print("5 symbols x 5yr history | next candle strictly after pattern close (no look-ahead)", flush=True)
    print("=" * 100, flush=True)

    rows = []   # (sym, pattern, label)
    for sym in SYM:
        df = load(sym)
        d = resample_30m(df)
        if len(d) < 30:
            continue
        vol_ma = d["volume"].rolling(20, min_periods=5).mean()
        c = d["close"].to_numpy(float)
        for i in range(1, len(d) - 1):  # need prev candle (i-1) and next candle (i+1)
            r = d.iloc[i]
            p = d.iloc[i - 1]
            v = float(r["volume"]) if pd.notna(r["volume"]) else None
            vm = float(vol_ma.iloc[i]) if pd.notna(vol_ma.iloc[i]) else None
            pats = detect_on_candle(r["open"], r["high"], r["low"], r["close"],
                                    p["open"], p["close"], p["high"], p["low"],
                                    vol=v, vol_ma=vm)
            for pat in pats:
                rows.append((sym, pat, label(c[i + 1] - c[i])))

    dfr = pd.DataFrame(rows, columns=["sym", "pattern", "label"])

    # baseline (unconditional next-candle direction)
    base = {k: int((dfr["label"] == k).sum()) for k in (-1, 0, 1)}
    n = len(dfr)
    base_up = base[1] / n
    base_dn = base[-1] / n
    print(f"\ntotal pattern observations: {n}", flush=True)
    print(f"baseline (any candle): up {base_up:.4f}  down {base_dn:.4f}  same {base[0]}", flush=True)

    print("\n--- PER PATTERN: sample count + what the NEXT candle did ---", flush=True)
    print(f"{'pattern':<24}{'n':>7}{'P(up)':>8}{'P(dn)':>8}{'imp.dir':>8}{'hit%':>8}", flush=True)
    print("-" * 100, flush=True)
    results = []
    for pat, g in dfr.groupby("pattern"):
        m = len(g)
        up = (g["label"] == 1).mean()
        dn = (g["label"] == -1).mean()
        d0 = PATTERN_DIR.get(pat, 0)
        # "hit" = next candle went the way the pattern implies
        if d0 == 1:
            hit = up
        elif d0 == -1:
            hit = dn
        else:
            hit = float("nan")
        results.append((pat, m, up, dn, d0, hit))
        print(f"{pat:<24}{m:>7}{up:>8.4f}{dn:>8.4f}{d0:>8}{hit:>8.4f}", flush=True)

    # per-symbol breakdown for the bear-side patterns that showed the edge
    print("\n--- BEAR-SIDE PATTERNS, per symbol (P next candle DOWN) ---", flush=True)
    bear = ["MARUBOZU BEAR", "BEAR ENGULFING", "SHOOTING STAR", "HIGH-VOL BEAR WICK",
            "HIGH-VOL BEAR REVERSAL"]
    print(f"{'pattern':<24}" + "".join(f"{s:>9}" for s in SYM), flush=True)
    for pat in bear:
        sub = dfr[dfr["pattern"] == pat]
        if len(sub) == 0:
            continue
        cells = []
        for s in SYM:
            g = sub[sub["sym"] == s]
            if len(g) < 30:
                cells.append(f"{'~':>9}")
            else:
                cells.append(f"{(g['label'] == -1).mean():>9.4f}")
        print(f"{pat:<24}" + "".join(cells), flush=True)

    print("\n--- BULL-SIDE PATTERNS, per symbol (P next candle UP) ---", flush=True)
    bull = ["MARUBOZU BULL", "BULL ENGULFING", "HAMMER", "HIGH-VOL BULL WICK",
            "HIGH-VOL BULL REVERSAL"]
    print(f"{'pattern':<24}" + "".join(f"{s:>9}" for s in SYM), flush=True)
    for pat in bull:
        sub = dfr[dfr["pattern"] == pat]
        if len(sub) == 0:
            continue
        cells = []
        for s in SYM:
            g = sub[sub["sym"] == s]
            if len(g) < 30:
                cells.append(f"{'~':>9}")
            else:
                cells.append(f"{(g['label'] == 1).mean():>9.4f}")
        print(f"{pat:<24}" + "".join(cells), flush=True)

    print("=" * 100, flush=True)
    print("VERDICT: pattern -> next-candle hit rates vs 50% coin flip above.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())