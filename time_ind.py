#!/usr/bin/env python3
"""time_ind.py — profile the per-labeler constructor cost (adx/ema/atr on
121k rows) to explain the gann slowdown."""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd
import numpy as np

import config
import indicators as ind


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "NQ_15min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    c = df["close"].to_numpy(float)

    for name, fn in [("atr", lambda: ind.atr(df, config.ATR_P)),
                     ("ema9", lambda: ind.ema(c, config.EMA_FAST)),
                     ("ema30", lambda: ind.ema(c, config.EMA_SLOW)),
                     ("adx", lambda: ind.adx(df, config.ADX_P))]:
        t0 = time.time()
        r = fn()
        print(f"{name}: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
