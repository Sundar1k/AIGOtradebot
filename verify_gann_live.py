#!/usr/bin/env python3
"""verify_gann_live.py — gann must still work on 3-min (no bundle) AND
use the ML bundle on 15-min. This mirrors the live supervisor path."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config
import pandas as pd

from strategies.gann_lane import GannAngleStrategy


def main():
    # 3-min: no gann bundle -> default proba 0.40, has_model False
    config.TIMEFRAME_MIN = 3
    g = GannAngleStrategy()
    df3 = pd.read_csv(os.path.join(HERE, "data", "NQ_3min.csv"))
    df3["time"] = pd.to_datetime(df3["datetime"], utc=True)
    df3 = df3.drop(columns=["datetime"]).reset_index(drop=True)
    print(f"3-min: model_path={os.path.basename(g.model_path())}  has_model={g.has_model()}")
    sig = g.detect(df3)
    if sig is not None:
        p, r = g.grade(df3, sig)
        print(f"3-min gann grade: proba={p} r_hat={r}  (expect 0.4, 0.0 — default path)")

    # 15-min: bundle exists -> ML grade
    config.TIMEFRAME_MIN = 15
    g15 = GannAngleStrategy()
    df15 = pd.read_csv(os.path.join(HERE, "data", "NQ_15min.csv"))
    df15["time"] = pd.to_datetime(df15["datetime"], utc=True)
    df15 = df15.drop(columns=["datetime"]).reset_index(drop=True)
    print(f"15-min: model_path={os.path.basename(g15.model_path())}  has_model={g15.has_model()}")
    sig15 = g15.detect(df15)
    if sig15 is not None:
        p, r = g15.grade(df15, sig15)
        print(f"15-min gann grade: proba={p:.3f} r_hat={r:.2f}  (ML bundle path)")
    else:
        print("15-min gann: no signal on last bar (fine)")
        from strategies.base import Signal
        from indicators import atr
        i = len(df15) - 1
        a = float(atr(df15, config.ATR_P)[i])
        entry = float(df15["close"].iloc[i])
        s2 = Signal("gann", 1, entry, entry - config.STOP_ATR * a, config.STOP_ATR * a, i, df15["time"].iloc[i])
        p, r = g15.grade(df15, s2)
        print(f"15-min gann forced grade: proba={p:.3f} r_hat={r:.2f}  (ML bundle path)")


if __name__ == "__main__":
    main()
