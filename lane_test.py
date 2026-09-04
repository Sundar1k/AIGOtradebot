import os
#!/usr/bin/env python3
"""lane_test.py — new lanes + 15-min retrain pre-check (Phase 4 protocol).

Lanes: ema (active) · orb (active) · supertrend (exists, inactive)
       · gann (NEW, strategies/gann_lane.py).
Configs at BOTH timeframes (3-min live data + 15-min resampled):
  BASE        ema + orb            (current live universe)
  +ST         base + supertrend
  +GANN       base + gann
  ALL         base + supertrend + gann

Rules identical to live: stop 0.5xATR(20), target 2R, out-of-sample
2025-26 only (the Phase 4 window). Any lane firing enters (approximates
best-signal-wins). No look-ahead. Raw engine — no ML, no veto.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import indicators as ind

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
TF = {"3min": None, "15min": "15min"}


def load(symbol: str, rule: str) -> pd.DataFrame:
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{symbol}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    if rule:
        d = df.set_index("time")
        df = d.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "volume": "sum"}).dropna().reset_index()
    return df


def simulate(df: pd.DataFrame, lanes: str):
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    t = df["time"]
    n = len(c)
    ef = ind.ema(c, config.EMA_FAST)
    es = ind.ema(c, config.EMA_SLOW)
    atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
    adx = np.asarray(ind.adx(df, config.ADX_P), dtype=float)
    oh, ol = ind.opening_range(df, config.ORB_BARS, config.ORB_OPEN_MIN, config.ORB_TZ)
    oh = np.asarray(oh, dtype=float)
    ol = np.asarray(ol, dtype=float)
    et_min = np.asarray(ind.et_minutes(df, config.ORB_TZ), dtype=float)
    st_line, st_dir = ind.supertrend(df, config.ST_PERIOD, config.ST_MULT)
    st_dir = np.asarray(st_dir, dtype=float)
    warm = max(120, config.ADX_P * 3, 2 * getattr(config, "GANN_SWING_K", 5) + 1)
    trades = []

    def settle(d, entry, risk, j0):
        stop = entry - d * risk
        tgt = entry + d * 2.0 * risk
        j = j0
        while j < n:
            if d > 0 and lo[j] <= stop:
                return -1.0, j
            if d < 0 and hi[j] >= stop:
                return -1.0, j
            if d > 0 and hi[j] >= tgt:
                return 2.0, j
            if d < 0 and lo[j] <= tgt:
                return 2.0, j
            j += 1
        return None, j

    def ema_sig(i):
        if not (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])
                and np.isfinite(adx[i])):
            return None
        if config.ADX_GATE and adx[i] < config.ADX_GATE:
            return None
        if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
            return 1
        if ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
            return -1
        return None

    def orb_sig(i):
        if not (np.isfinite(oh[i]) and np.isfinite(oh[i - 1])
                and np.isfinite(ol[i]) and np.isfinite(ol[i - 1])
                and np.isfinite(adx[i])):
            return None
        if et_min[i] >= config.ORB_CLOSE_MIN or adx[i] < config.ORB_ADX_GATE:
            return None
        if c[i - 1] <= oh[i - 1] and c[i] > oh[i]:
            return 1
        if c[i - 1] >= ol[i - 1] and c[i] < ol[i]:
            return -1
        return None

    def st_sig(i):
        if i < 1 or not (np.isfinite(st_dir[i]) and np.isfinite(st_dir[i - 1])):
            return None
        if st_dir[i] == st_dir[i - 1]:
            return None
        return int(st_dir[i])

    def gann_sig(i):
        if i < 2 * getattr(config, "GANN_SWING_K", 5) + 1:
            return None
        if not (np.isfinite(atr[i]) and np.isfinite(atr[i - 1]) and atr[i] > 0):
            return None
        k = getattr(config, "GANN_SWING_K", 5)
        m = getattr(config, "GANN_SLOPE_MULT", 0.5)
        lo_ = lo
        hi_ = hi
        low_i, high_i = -1, -1
        for j in range(i - k - 1, k - 1, -1):
            if low_i < 0 and lo_[j] == lo_[j - k:j + k + 1].min():
                low_i = j
            if high_i < 0 and hi_[j] == hi_[j - k:j + k + 1].max():
                high_i = j
            if low_i >= 0 and high_i >= 0:
                break
        if low_i < 0 or high_i < 0:
            return None
        up = lo_[low_i] + m * atr[i] * (i - low_i)
        dn = hi_[high_i] - m * atr[i] * (i - high_i)
        p_up = c[i - 1] > (lo_[low_i] + m * atr[i - 1] * (i - 1 - low_i))
        p_dn = c[i - 1] < (hi_[high_i] - m * atr[i - 1] * (i - 1 - high_i))
        c_up = c[i] > up
        c_dn = c[i] < dn
        if p_up and not c_up and not c_dn:
            return -1
        if p_dn and not c_dn and not c_up:
            return 1
        if not p_up and not p_dn:
            if c_up:
                return 1
            if c_dn:
                return -1
        return None

    i = warm
    while i < n - 1:
        sig = None
        if "ema" in lanes:
            sig = ema_sig(i)
        if sig is None and "orb" in lanes:
            sig = orb_sig(i)
        if sig is None and "st" in lanes:
            sig = st_sig(i)
        if sig is None and "gann" in lanes:
            sig = gann_sig(i)
        if sig is not None and np.isfinite(atr[i]) and atr[i] > 0:
            r, jx = settle(sig, c[i], 0.5 * atr[i], i + 1)
            if r is not None:
                trades.append((sig, r, t.iloc[i]))
                i = jx + 1
                continue
        i += 1
    return trades


def stats(rows):
    if not rows:
        return dict(n=0, wr=0.0, avg=0.0, sm=0.0, pf=0.0)
    rs = np.array([r[1] for r in rows])
    n = len(rs)
    wr = float(np.sum(rs > 0)) / n
    avg = float(np.mean(rs))
    g_win = float(np.sum(rs[rs > 0]))
    g_loss = -float(np.sum(rs[rs < 0]))
    pf = g_win / g_loss if g_loss > 0 else float("inf")
    return dict(n=n, wr=wr, avg=avg, sm=float(np.sum(rs)), pf=pf)


def main():
    print("=== LANE TEST — supertrend + NEW gann lane · 3-min AND 15-min (OOS 2025-26) ===", flush=True)
    print("=" * 96, flush=True)
    configs = [("BASE", "ema,orb"), ("+ST", "ema,orb,st"), ("+GANN", "ema,orb,gann"),
               ("ALL", "ema,orb,st,gann")]
    for tf_label, rule in TF.items():
        print(f"\n--- {tf_label.upper()} candles ---", flush=True)
        print(f"{'config':<8} {'n':>7} {'WR':>7} {'avgR':>8} {'sumR':>9} {'PF':>6}", flush=True)
        print("-" * 60, flush=True)
        base = None
        for label, lanes in configs:
            all_tr = []
            for s in SYMBOLS:
                df = load(s, rule)
                tr = simulate(df, lanes)
                all_tr += [x for x in tr if x[2] >= OOS_START]
            st = stats(all_tr)
            if base is None:
                base = st
                tag = "baseline"
            else:
                tag = ("✓ beats" if st["n"] >= 100 and st["avg"] > base["avg"]
                       and st["pf"] > base["pf"] else "✗")
            print(f"{label:<8} {st['n']:>7} {st['wr']:>7.1%} {st['avg']:>+8.3f} "
                  f"{st['sm']:>+9.0f} {st['pf']:>6.2f}  {tag}", flush=True)
    print("\n" + "=" * 96, flush=True)
    print("Accept rule: n>=100 AND avgR > BASE AND PF > BASE (out-of-sample).", flush=True)
    print("Raw engine only — the ML+veto funnel sits on top in live.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
