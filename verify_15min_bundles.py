#!/usr/bin/env python3
"""verify_15min_bundles.py — load the 15-min bundles through the bot's
actual strategy model_path() convention (TIMEFRAME_MIN=15) and sanity-
check a grade() call on a real 15-min bar."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
config.TIMEFRAME_MIN = 15

import pandas as pd
import numpy as np

from strategies.ema_cross import EmaCrossStrategy
from strategies.gann_lane import GannAngleStrategy
from strategies.supertrend import SuperTrendStrategy
from strategies.orb import OrbStrategy


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "NQ_15min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop(columns=["datetime"]).reset_index(drop=True)
    print(f"bars: {len(df)}  cols: {list(df.columns)}")

    for name, strat in [("ema", EmaCrossStrategy()),
                        ("gann", GannAngleStrategy()),
                        ("st", SuperTrendStrategy()),
                        ("orb", OrbStrategy())]:
        try:
            p = strat.model_path()
            print(f"{name}: model_path={os.path.basename(p)}  has_model={strat.has_model()}")
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__}: {e}")

    # grade() sanity: ema should fire + grade on a 15-min bar with a bundle
    ema = EmaCrossStrategy()
    if ema.has_model():
        sig = ema.detect(df)
        if sig is not None:
            proba, r_hat = ema.grade(df, sig)
            print(f"ema grade on last bar: sig={sig.direction} proba={proba:.3f} r_hat={r_hat:.2f}")
        else:
            print("ema: no signal on last bar (fine — detect() returned None)")
        # force a grade with a synthetic signal to test the bundle path
        from strategies.base import Signal
        import config as cfg
        from indicators import atr
        i = len(df) - 1
        a = float(atr(df, cfg.ATR_P)[i])
        entry = float(df["close"].iloc[i])
        sig2 = Signal("ema", 1, entry, entry - cfg.STOP_ATR * a, cfg.STOP_ATR * a, i, df["time"].iloc[i])
        proba, r_hat = ema.grade(df, sig2)
        print(f"ema forced grade: proba={proba:.3f} r_hat={r_hat:.2f}  (feat path OK)")


if __name__ == "__main__":
    main()
