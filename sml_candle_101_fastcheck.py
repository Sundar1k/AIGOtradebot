import os
#!/usr/bin/env python3
"""sml_candle_101_fastcheck.py — 5yr / 3-min candle -> next-candle -1/0/1 fast check.

User's spec:
  - 5 symbols, exactly 5 years back from today, 3-min candles only
  - label per candle: next candle close vs current close
        +1 = above, 0 = same level (within +-1bp), -1 = below
  - model "sees" a small window of recent candles -> predicts next -1/0/1

Fast check = XGBoost (minutes), to measure whether there is ANY learnable
signal before committing to a full QLoRA SLM run. Point-in-time split, no
look-ahead.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYM = ["NQ", "ES", "RTY", "YM", "GC"]
FLAT_EPS = 0.0001                     # 1 bp = "same level"
TRAIN_END = pd.Timestamp("2026-06-01", tz="UTC")
VAL_END   = pd.Timestamp("2026-07-01", tz="UTC")
TEST_START = pd.Timestamp("2026-08-01", tz="UTC")


def load(sym):
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{sym}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


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


def build(df):
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    a = atr(df, 14)
    logc = np.log(c)
    rng = h - l
    rng = np.where(rng > 0, rng, np.nan)

    F = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10, 20):          # recent returns (window of past candles)
        F[f"ret{k}"] = pd.Series(logc).diff(k).to_numpy()
    body = np.abs(c - o)
    F["body_ratio"] = body / rng            # candle body vs range (doji vs marubozu)
    F["upper_wick"] = (h - np.maximum(o, c)) / rng
    F["lower_wick"] = (np.minimum(o, c) - l) / rng
    F["rsi"] = rsi(c, 14)
    F["vol_norm"] = a / c
    F["ema_spread"] = (pd.Series(c).ewm(span=9, adjust=False).mean().to_numpy()
                       - pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()) / a
    F["vol_chg"] = pd.Series(np.log(v + 1.0)).diff(5).to_numpy()

    # label: next candle close vs current close
    nxt = np.full(len(c), np.nan)
    nxt[:-1] = c[1:]
    ret_next = (nxt - c) / c
    y = np.full(len(c), np.nan)
    y[ret_next > FLAT_EPS] = 1
    y[ret_next < -FLAT_EPS] = -1
    y[(ret_next >= -FLAT_EPS) & (ret_next <= FLAT_EPS)] = 0
    F["y"] = y
    return F


def main():
    print("=== 3-MIN CANDLE -> NEXT CANDLE -1/0/1  (5yr, 5 symbols) FAST CHECK ===", flush=True)
    print(f"flat = +-{FLAT_EPS*100:.2f}% (1bp) | train<{TRAIN_END.date()} val=Jun test>=Aug", flush=True)
    print("=" * 90, flush=True)

    from xgboost import XGBClassifier

    frames = []
    for sym in SYM:
        df = load(sym)
        F = build(df)
        F["sym"] = sym
        F["time"] = df["time"]
        frames.append(F)
    allf = pd.concat(frames, ignore_index=True)
    allf = allf.dropna(subset=list(allf.columns))

    feat = [c for c in allf.columns if c not in ("sym", "time", "y")]
    t = allf["time"]
    tr_m = t < TRAIN_END
    va_m = (t >= TRAIN_END) & (t < VAL_END)
    te_m = t >= TEST_START

    def dist(y):
        return {k: int((y == k).sum()) for k in (-1, 0, 1)}

    print(f"\nsamples train={int(tr_m.sum())} val={int(va_m.sum())} test={int(te_m.sum())}", flush=True)
    yte = allf.loc[te_m, "y"].to_numpy(int)
    print(f"test label balance: down={dist(yte)[-1]}  same={dist(yte)[0]}  up={dist(yte)[1]}", flush=True)

    Xtr = allf.loc[tr_m, feat].to_numpy(float); ytr = allf.loc[tr_m, "y"].to_numpy(int)
    Xva = allf.loc[va_m, feat].to_numpy(float); yva = allf.loc[va_m, "y"].to_numpy(int)
    Xte = allf.loc[te_m, feat].to_numpy(float)

    mp = {-1: 0, 0: 1, 1: 2}
    ytr_m = np.array([mp[y] for y in ytr]); yva_m = np.array([mp[y] for y in yva]); yte_m = np.array([mp[y] for y in yte])

    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9, random_state=42,
                      n_jobs=-1, tree_method="hist", eval_metric="mlogloss",
                      early_stopping_rounds=30)
    m.fit(Xtr, ytr_m, eval_set=[(Xva, yva_m)], verbose=False)

    pte_m = m.predict(Xte)
    inv = {0: -1, 1: 0, 2: 1}
    pte = np.array([inv[p] for p in pte_m])

    acc3 = float((pte == yte).mean())
    maj = max(dist(yte).values()) / len(yte)
    t2 = yte != 0
    acc2 = float((pte[t2] == yte[t2]).mean()) if t2.sum() else float("nan")

    print(f"\n3-class acc : {acc3:.4f}   (majority baseline = {maj:.4f})", flush=True)
    print(f"2-class acc : {acc2:.4f}   n={int(t2.sum())}   (coin flip = 0.5000)", flush=True)

    print("\n--- per symbol 2-class (up/down) ---", flush=True)
    syms = allf.loc[te_m, "sym"].to_numpy()
    for s in SYM:
        msk = syms == s
        sub2 = yte[msk] != 0
        if sub2.sum() == 0:
            continue
        print(f"  {s:<4} {((pte[msk][sub2] == yte[msk][sub2]).mean()):.4f}  n={int(sub2.sum())}", flush=True)

    # feature importance (what did it lean on, if anything)
    imp = sorted(zip(feat, m.feature_importances_), key=lambda x: -x[1])[:5]
    print("\n--- top features the model used ---", flush=True)
    for f, w in imp:
        print(f"  {f:<14} {w:.4f}", flush=True)

    print("=" * 90, flush=True)
    if acc2 < 0.53:
        v = "NO LEARNABLE SIGNAL (coin flip) — full SLM run not worth the GPU"
    elif acc2 < 0.55:
        v = "MARGINAL — not enough for a tradeable edge"
    else:
        v = "SIGNAL PRESENT — proceed to full SLM (QLoRA) training"
    print(f"VERDICT: 2-class {acc2:.4f} -> {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())