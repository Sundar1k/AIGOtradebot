#!/usr/bin/env python3
"""time_gann_scan.py — benchmark the gann _scan loop on 15-min bars."""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import config
import indicators as ind


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "NQ_15min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    c = df["close"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
    n = len(c)
    k = getattr(config, "GANN_SWING_K", 5)
    t0 = time.time()
    cnt = 0
    for i in range(max(120, 2 * k + 1), min(n, 20000)):
        if not (np.isfinite(atr[i]) and np.isfinite(atr[i - 1]) and atr[i] > 0):
            continue
        low_i, high_i = -1, -1
        for j in range(i - k - 1, k - 1, -1):
            if low_i < 0 and lo[j] == lo[j - k:j + k + 1].min():
                low_i = j
            if high_i < 0 and hi[j] == hi[j - k:j + k + 1].max():
                high_i = j
        cnt += 1
    dt = time.time() - t0
    per_sym = dt * n / cnt / 60
    print(f"{cnt} bars in {dt:.1f}s -> {per_sym:.1f} min/symbol, "
          f"{5 * per_sym:.1f} min for 5 symbols")


if __name__ == "__main__":
    main()
