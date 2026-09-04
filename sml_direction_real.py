#!/usr/bin/env python3
"""sml_direction_real.py — REAL -1/0/1 next-candle direction test on actual 3-min data.

Labels (exactly as the user described):
    +1  = next 3-min candle closes ABOVE  current close (up / green)
     0  = next 3-min candle closes AT SAME level (tie — essentially never at 3-min)
    -1  = next 3-min candle closes BELOW  current close (down / red)

Causal features (only data up to bar i). Point-in-time split, no look-ahead.
Model: XGBoost. Money metric = 2-class up/down accuracy vs the 50% coin flip.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYM = ["NQ", "ES", "RTY", "YM", "GC"]
TRAIN_END   = pd.Timestamp("2026-06-01", tz="UTC")
VAL_END     = pd.Timestamp("2026-07-01", tz="UTC")   # June block = val
TEST_START  = pd.Timestamp("2026-08-01", tz="UTC")   # most-recent Aug block = blind test


def load(sym):
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().to_numpy()


def rsi(c, period=14):
    c = pd.Series(c)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).to_numpy()


def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def build_features(df):
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    a = atr(df, 14)
    logc = np.log(c)
    rets = {}
    for k in (1, 3, 5, 10, 20):
        rets[k] = pd.Series(logc).diff(k).to_numpy()
    e9, e20 = ema(c, 9), ema(c, 20)
    return pd.DataFrame({
        "ret1": rets[1], "ret3": rets[3], "ret5": rets[5],
        "ret10": rets[10], "ret20": rets[20],
        "ema_spread": (e9 - e20) / (a + 1e-9),
        "rsi": rsi(c, 14),
        "vol_norm": a / (c + 1e-9),
        "vol_chg": pd.Series(np.log(v + 1.0)).diff(5).to_numpy(),
    })


def build_labels(df):
    c = df["close"].to_numpy(float)
    y = np.zeros(len(c), dtype=int)
    d = np.zeros(len(c))
    d[:-1] = c[1:] - c[:-1]
    y[d > 0] = 1
    y[d < 0] = -1
    # last bar has no next bar -> label it -999 to drop
    y[-1] = -999
    return y


def main():
    print("=== REAL -1/0/1 next-3min-candle classifier (5 symbols) ===", flush=True)
    print(f"label: +1 up / 0 same-level / -1 down | train<2026-06-01, val=Jun, test>=2026-08-01", flush=True)
    print("=" * 90, flush=True)

    from xgboost import XGBClassifier

    frames = []
    for sym in SYM:
        df = load(sym)
        F = build_features(df)
        F["sym"] = sym
        F["time"] = df["time"]
        F["y"] = build_labels(df)
        frames.append(F)
    allf = pd.concat(frames, ignore_index=True)
    allf = allf[allf["y"] != -999]
    allf = allf.dropna(subset=list(allf.columns))

    feat_cols = ["ret1", "ret3", "ret5", "ret10", "ret20", "ema_spread", "rsi", "vol_norm", "vol_chg"]
    t = allf["time"]
    tr_m = t < TRAIN_END
    va_m = (t >= TRAIN_END) & (t < VAL_END)
    te_m = t >= TEST_START

    Xtr, ytr = allf.loc[tr_m, feat_cols].to_numpy(float), allf.loc[tr_m, "y"].to_numpy(int)
    Xva, yva = allf.loc[va_m, feat_cols].to_numpy(float), allf.loc[va_m, "y"].to_numpy(int)
    Xte, yte = allf.loc[te_m, feat_cols].to_numpy(float), allf.loc[te_m, "y"].to_numpy(int)

    def dist(y):
        return {k: int((y == k).sum()) for k in (-1, 0, 1)}

    print(f"\nsamples  train={len(ytr)}  val={len(yva)}  test={len(yte)}", flush=True)
    print(f"label counts (test): down={dist(yte)[-1]}  same={dist(yte)[0]}  up={dist(yte)[1]}", flush=True)

    # map {-1,0,1} -> {0,1,2} for XGBoost
    mp = {-1: 0, 0: 1, 1: 2}
    ytr_m = np.array([mp[y] for y in ytr]); yva_m = np.array([mp[y] for y in yva]); yte_m = np.array([mp[y] for y in yte])

    m = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, random_state=42,
                      n_jobs=-1, tree_method="hist", eval_metric="mlogloss",
                      early_stopping_rounds=30)
    m.fit(Xtr, ytr_m, eval_set=[(Xva, yva_m)], verbose=False)

    pte_m = m.predict(Xte)
    inv = {0: -1, 1: 0, 2: 1}
    pte = np.array([inv[p] for p in pte_m])

    # ---- 3-class accuracy vs majority ----
    acc3 = float((pte == yte).mean())
    maj = max(dist(yte).values()) / len(yte)

    # ---- 2-class up/down (the money metric) ----
    t2 = yte != 0
    acc2 = float((pte[t2] == yte[t2]).mean()) if t2.sum() else float("nan")

    print(f"\n3-class acc  : {acc3:.4f}   (majority baseline = {maj:.4f})", flush=True)
    print(f"2-class acc  : {acc2:.4f}   n={int(t2.sum())}   (coin flip = 0.5000)", flush=True)

    print("\n--- per symbol (2-class up/down) ---", flush=True)
    syms = allf.loc[te_m, "sym"].to_numpy()
    for s in SYM:
        msk = syms == s
        sub2 = yte[msk] != 0
        if sub2.sum() == 0:
            continue
        a = (pte[msk][sub2] == yte[msk][sub2]).mean()
        print(f"  {s:<4} 2-class {a:.4f}   n={int(sub2.sum())}", flush=True)

    print("=" * 90, flush=True)
    if acc2 < 0.53:
        v = "COIN FLIP — next-candle direction is not predictable (matches 6 prior tests)"
    elif acc2 < 0.55:
        v = "MARGINAL"
    else:
        v = "REAL EDGE >55%"
    print(f"VERDICT: 2-class up/down = {acc2:.4f} -> {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())