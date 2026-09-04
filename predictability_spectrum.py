#!/usr/bin/env python3
"""predictability_spectrum.py — at which timeframe does price direction
become predictable?

Pure statistics, no models: for each horizon (15m, 1h, 4h, daily,
weekly) on each symbol, measure
  1. lag-1 autocorrelation of returns        (classic "is it a random walk")
  2. momentum hit-rate: sign(past N-bar return) == sign(next N-bar return)
     using NON-OVERLAPPING windows (independent samples, honest p-values).

Hit-rate > 50% means the past predicts the future at that horizon.
SE = 0.5/sqrt(n) — the noise band around 50%.
"""
import sys
import numpy as np
import pandas as pd

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
HORIZONS = ["15min", "1h", "4h", "1D", "1W"]


def load(symbol: str) -> pd.Series:
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{symbol}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("time")["close"]


def spectrum(closes: pd.Series, rule: str) -> dict:
    c = closes.resample(rule).last().dropna()
    ret = c.pct_change().dropna().to_numpy()
    # 1) lag-1 autocorrelation of returns (random-walk test)
    ac = float(np.corrcoef(ret[:-1], ret[1:])[0, 1]) if len(ret) > 50 else float("nan")
    # 2) momentum hit-rate: sign(ret_t) == sign(ret_{t+1})
    #    consecutive resampled returns are non-overlapping windows.
    prod = np.sign(ret[:-1]) * np.sign(ret[1:])
    hits = int(np.sum(prod > 0))
    n_h = len(prod)
    return dict(ac=ac, hr=hits / n_h if n_h else float("nan"), n=n_h)


def main():
    print("=== PREDICTABILITY SPECTRUM — 5 years of your data ===", flush=True)
    print("horizon : autocorr | momentum hit-rate | n(samples) | verdict", flush=True)
    print("-" * 88, flush=True)
    for rule in HORIZONS:
        rows = []
        for s in SYMBOLS:
            rows.append(spectrum(load(s), rule))
        acs = [r["ac"] for r in rows if np.isfinite(r["ac"])]
        hrs = [r["hr"] for r in rows if np.isfinite(r["hr"])]
        ns = [r["n"] for r in rows if r["n"] > 0]
        if not hrs:
            print(f"{rule:>6} : no data", flush=True)
            continue
        hr = float(np.mean(hrs))
        n = int(np.min(ns))
        se = 0.5 / np.sqrt(n)
        ac = float(np.mean(acs)) if acs else float("nan")
        z = (hr - 0.5) / se
        if abs(z) > 3:
            direction = "REVERSAL" if hr < 0.5 else "MOMENTUM"
            verdict = f"REAL {direction} (z={z:+.0f})"
        elif abs(z) > 2:
            direction = "reversal" if hr < 0.5 else "momentum"
            verdict = f"marginal {direction} (z={z:+.1f})"
        else:
            verdict = f"noise (z={z:+.1f})"
        per_sym = " ".join(f"{s[:2]}:{r['hr']:.0%}" for s, r in zip(SYMBOLS, rows))
        print(f"{rule:>6} : {ac:+.3f}   | {hr:>6.1%} ±{1.96*se:.1%} | n={n:>5} | {verdict} | {per_sym}", flush=True)
    print("-" * 88, flush=True)
    print("Read: hit-rate above 50%+2SE means past return predicts next return.", flush=True)
    print("z = (hitrate - 0.5)/SE ; z>3 = strong evidence of real predictability.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
