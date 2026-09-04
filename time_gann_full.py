#!/usr/bin/env python3
"""time_gann_full.py — full-length gann _scan benchmark + vectorized option."""
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


def scan_slow(lo, hi, atr, k, m, n):
    """The current GannChronos._fired loop (per-bar Python swing search)."""
    sigs = []
    for i in range(2 * k + 1, n):
        if not (np.isfinite(atr[i]) and np.isfinite(atr[i - 1]) and atr[i] > 0):
            continue
        low_i, high_i = -1, -1
        for j in range(i - k - 1, k - 1, -1):
            if low_i < 0 and lo[j] == lo[j - k:j + k + 1].min():
                low_i = j
            if high_i < 0 and hi[j] == hi[j - k:j + k + 1].max():
                high_i = j
        if low_i >= 0 and high_i >= 0:
            sigs.append(i)
    return sigs


def scan_fast(lo, hi, k, n):
    """Vectorized fractal swings: lo[i] is a swing low iff it's the min of
    its ±k window; same for hi. O(n) with numpy rolling argmin."""
    from numpy.lib.stride_tricks import sliding_window_view
    lows = np.zeros(n, bool)
    highs = np.zeros(n, bool)
    # padding: use the first/last valid window value at edges
    lo_pad = np.concatenate([np.full(k, lo[0]), lo, np.full(k, lo[-1])])
    hi_pad = np.concatenate([np.full(k, hi[0]), hi, np.full(k, hi[-1])])
    wlo = sliding_window_view(lo_pad, 2 * k + 1)
    whi = sliding_window_view(hi_pad, 2 * k + 1)
    for i in range(n):
        win_lo = wlo[i]
        win_hi = whi[i]
        lows[i] = (lo[i] == win_lo.min())
        highs[i] = (hi[i] == win_hi.max())
    return lows, highs


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "NQ_15min.csv"))
    lo = df["low"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
    n = len(lo)
    k = getattr(config, "GANN_SWING_K", 5)

    t0 = time.time()
    sigs = scan_slow(lo, hi, atr, k, 0.5, n)
    print(f"slow scan: {len(sigs)} sigs in {time.time()-t0:.1f}s")

    t0 = time.time()
    lows, highs = scan_fast(lo, hi, k, n)
    print(f"fast swing detect: {lows.sum()} lows / {highs.sum()} highs in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
