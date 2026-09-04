import os
#!/usr/bin/env python3
"""sml_candle_101_balanced.py — BALANCED 3-class version of the fast check.

The 1bp "flat" band made 78.6% of candles "flat", which collapsed the model.
This version sets "flat" to the middle third of |next move|, so the three
classes are ~equal — the fair way to ask "can a model learn up/flat/down?"
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYM = ["NQ", "ES", "RTY", "YM", "GC"]
TRAIN_END = pd.Timestamp("2026-06-01", tz="UTC")
VAL_END   = pd.Timestamp("2026-07-01", tz="UTC")
TEST_START = pd.Timestamp("2026-08-01", tz="UTC")


def load(sym):
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def rsi(c, period=14):
    c = pd.Series(c); d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).to_numpy()


def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def build(df):
    c = df["close"].to_numpy(float); o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    a = atr(df, 14)
    logc = np.log(c)
    rng = np.where(h - l > 0, h - l, np.nan)
    F = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10, 20):
        F[f"ret{k}"] = pd.Series(logc).diff(k).to_numpy()
    body = np.abs(c - o)
    F["body_ratio"] = body / rng
    F["upper_wick"] = (h - np.maximum(o, c)) / rng
    F["lower_wick"] = (np.minimum(o, c) - l) / rng
    F["rsi"] = rsi(c, 14)
    F["vol_norm"] = a / c
    F["ema_spread"] = (pd.Series(c).ewm(span=9, adjust=False).mean().to_numpy()
                       - pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()) / a
    F["vol_chg"] = pd.Series(np.log(v + 1.0)).diff(5).to_numpy()
    nxt = np.full(len(c), np.nan); nxt[:-1] = c[1:]
    F["ret_next"] = (nxt - c) / c
    return F


def main():
    print("=== BALANCED 3-class: 3-min candle -> next -1/0/1 (5yr, 5 symbols) ===", flush=True)
    print("flat = middle third of |next move| (classes ~equal) | train<Jun val=Jun test>=Aug", flush=True)
    print("=" * 90, flush=True)

    from xgboost import XGBClassifier

    frames = []
    for sym in SYM:
        df = load(sym)
        F = build(df)
        F["sym"] = sym; F["time"] = df["time"]
        frames.append(F)
    allf = pd.concat(frames, ignore_index=True)

    # flat threshold from TRAIN data only (no look-ahead)
    tr_mask = allf["time"] < TRAIN_END
    thr = float(np.nanpercentile(np.abs(allf.loc[tr_mask, "ret_next"]), 33.3))
    print(f"\n'flat' threshold (33rd pct of |next move|) = {thr*100:.4f}%  ({thr*10000:.1f} bp)", flush=True)

    rn = allf["ret_next"].to_numpy()
    y = np.where(rn > thr, 1, np.where(rn < -thr, -1, 0))
    allf["y"] = y
    allf = allf.dropna(subset=[c for c in allf.columns if c not in ("y",)])[allf["y"].notna()]

    feat = [c for c in allf.columns if c not in ("sym", "time", "y", "ret_next")]
    t = allf["time"]
    trm = t < TRAIN_END; vam = (t >= TRAIN_END) & (t < VAL_END); tem = t >= TEST_START

    def dist(a):
        return {k: int((a == k).sum()) for k in (-1, 0, 1)}

    yte = allf.loc[tem, "y"].to_numpy(int)
    print(f"test balance: down={dist(yte)[-1]} flat={dist(yte)[0]} up={dist(yte)[1]}", flush=True)

    Xtr = allf.loc[trm, feat].to_numpy(float); ytr = allf.loc[trm, "y"].to_numpy(int)
    Xva = allf.loc[vam, feat].to_numpy(float); yva = allf.loc[vam, "y"].to_numpy(int)
    Xte = allf.loc[tem, feat].to_numpy(float)

    mp = {-1: 0, 0: 1, 1: 2}
    ytr_m = np.array([mp[y] for y in ytr]); yva_m = np.array([mp[y] for y in yva]); yte_m = np.array([mp[y] for y in yte])
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, random_state=42,
                      n_jobs=-1, tree_method="hist", eval_metric="mlogloss",
                      early_stopping_rounds=30)
    m.fit(Xtr, ytr_m, eval_set=[(Xva, yva_m)], verbose=False)
    inv = {0: -1, 1: 0, 2: 1}
    pte = np.array([inv[p] for p in m.predict(Xte)])

    acc3 = float((pte == yte).mean())
    t2 = yte != 0
    acc2 = float((pte[t2] == yte[t2]).mean()) if t2.sum() else float("nan")
    # confusion matrix
    cm = np.zeros((3, 3), int)
    for a, b in zip(yte, pte):
        cm[mp[int(a)]][mp[int(b)]] += 1
    print(f"\n3-class acc : {acc3:.4f}   (balanced baseline = 0.3333)", flush=True)
    print(f"2-class acc : {acc2:.4f}   n={int(t2.sum())}   (coin flip = 0.5000)", flush=True)
    print("\nconfusion (rows=true, cols=pred)  [-1,0,1]:", flush=True)
    print(cm, flush=True)

    print("=" * 90, flush=True)
    if acc2 >= 0.55 or acc3 >= 0.40:
        v = "SIGNAL PRESENT — proceed to full SLM"
    elif acc2 >= 0.53:
        v = "MARGINAL"
    else:
        v = "NO LEARNABLE SIGNAL (direction is a coin flip)"
    print(f"VERDICT: 3-class {acc3:.4f} / 2-class {acc2:.4f} -> {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())