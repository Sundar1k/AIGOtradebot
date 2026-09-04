import os
#!/usr/bin/env python3
"""ema_orb_conditional.py — WHERE do EMA/ORB signals actually win?

Searches for the conditions (factors) where EMA and ORB signals reach
high winrates, using the bot's exact rules (stop 0.5xATR(20), target 2R).
Honest protocol:
  - in-sample  = 2021-01-01 .. 2024-12-31 (search here)
  - out-of-sample = 2025-01-01 .. now       (test the found cells)
  - only cells with n>=100 in-sample are candidates
  - NOTE: at 2R/1R breakeven winrate is 33% — cells above ~40% that
    HOLD out-of-sample are genuinely profitable raw signals.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import indicators as ind

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
BREAKEVEN = 1.0 / 3.0          # 2R/1R


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{symbol}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def simulate(df: pd.DataFrame):
    """Run BOTH lanes (ema + orb) with the bot's exact rules. Returns
    list of dicts: strategy, side, r, et_hour, rth, atr_pct, adx, dow, dist."""
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    t = df["time"]
    ef = ind.ema(c, config.EMA_FAST)
    es = ind.ema(c, config.EMA_SLOW)
    atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
    adx = np.asarray(ind.adx(df, config.ADX_P), dtype=float)
    oh, ol = ind.opening_range(df, config.ORB_BARS, config.ORB_OPEN_MIN, config.ORB_TZ)
    oh = np.asarray(oh, dtype=float)
    ol = np.asarray(ol, dtype=float)
    et_min = np.asarray(ind.et_minutes(df, config.ORB_TZ), dtype=float)
    utc_h = t.dt.hour.to_numpy()
    dow = t.dt.dayofweek.to_numpy()
    n = len(c)
    # ATR percentile per symbol (vol regime)
    atr_ok = atr[np.isfinite(atr) & (atr > 0)]
    p33, p66 = np.percentile(atr_ok, [33, 66]) if len(atr_ok) else (0, 1)

    warm = max(80, config.ADX_P * 3)
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

    def rec(strategy, side, r, i, dist):
        trades.append(dict(strategy=strategy, side=side, r=r, t=t.iloc[i],
                           et_hour=(utc_h[i] - 4) % 24,        # ET approx (EDT)
                           rth=1 if 9.5 <= (utc_h[i] - 4) % 24 < 16 else 0,
                           atr_pct=1 if atr[i] > p66 else (0 if atr[i] < p33 else 2),
                           adx=int(adx[i]) if np.isfinite(adx[i]) else 0,
                           dow=int(dow[i]), dist=dist))

    i = warm
    while i < n - 1:
        # ---- EMA lane ----
        if (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            if adx[i] >= config.ADX_GATE:
                if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
                    d, risk = 1, 0.5 * atr[i]
                    r, jx = settle(d, c[i], risk, i + 1)
                    if r is not None:
                        rec("ema", d, r, i, (c[i] - es[i]) / atr[i])
                        i = jx + 1
                        continue
                elif ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
                    d, risk = -1, 0.5 * atr[i]
                    r, jx = settle(d, c[i], risk, i + 1)
                    if r is not None:
                        rec("ema", d, r, i, (c[i] - es[i]) / atr[i])
                        i = jx + 1
                        continue
        # ---- ORB lane ----
        if (np.isfinite(oh[i]) and np.isfinite(oh[i - 1])
                and np.isfinite(ol[i]) and np.isfinite(ol[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i]) and atr[i] > 0):
            if et_min[i] < config.ORB_CLOSE_MIN and adx[i] >= config.ORB_ADX_GATE:
                if c[i - 1] <= oh[i - 1] and c[i] > oh[i]:
                    d, risk = 1, 0.5 * atr[i]
                    r, jx = settle(d, c[i], risk, i + 1)
                    if r is not None:
                        rec("orb", d, r, i, (c[i] - oh[i]) / atr[i])
                        i = jx + 1
                        continue
                elif c[i - 1] >= ol[i - 1] and c[i] < ol[i]:
                    d, risk = -1, 0.5 * atr[i]
                    r, jx = settle(d, c[i], risk, i + 1)
                    if r is not None:
                        rec("orb", d, r, i, (c[i] - ol[i]) / atr[i])
                        i = jx + 1
                        continue
        i += 1
    return trades


FACTORS = {
    "side": lambda t: "LONG" if t["side"] > 0 else "SHORT",
    "et_hour": lambda t: f"{int(t['et_hour']):02d}h ET",
    "session": lambda t: "RTH" if t["rth"] else "OVERNIGHT",
    "vol(ATR pct)": lambda t: {0: "low", 2: "mid", 1: "high"}[t["atr_pct"]],
    "adx": lambda t: f"{max(18, min(50, (t['adx'] // 10) * 10))}-{max(18, min(50, (t['adx'] // 10) * 10)) + 10}",
    "day": lambda t: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][t["dow"]],
    "dist(ATR)": lambda t: (f"{abs(t['dist']):.1f}-{abs(t['dist']) + 0.5:.1f}"
                            if abs(t["dist"]) < 3 else "3.0+"),
}


def wr(rows):
    if rows is None or len(rows) == 0:
        return None, 0
    rs = rows["r"].to_numpy()
    return float(np.sum(rs > 0)) / len(rs), len(rs)


def main():
    print("=== EMA + ORB CONDITIONAL WINRATE SEARCH (5y) ===", flush=True)
    print("in-sample 2021-24 (search) · out-of-sample 2025-26 (test) · breakeven at 2R = 33%", flush=True)
    print("=" * 108, flush=True)
    all_trades = []
    for s in SYMBOLS:
        all_trades += simulate(load(s))
    df = pd.DataFrame(all_trades)
    print(f"  total trades: {len(df)} | EMA {len(df[df.strategy=='ema'])} | "
          f"ORB {len(df[df.strategy=='orb'])}", flush=True)
    ins = df[df["t"] < OOS_START]
    oos = df[df["t"] >= OOS_START]
    print(f"  in-sample: {len(ins)} · out-of-sample: {len(oos)}", flush=True)

    for strat in ("ema", "orb"):
        si, so = ins[ins.strategy == strat], oos[oos.strategy == strat]
        print(f"\n{'='*108}", flush=True)
        print(f"  STRATEGY: {strat.upper()} — winrate by factor (in-sample → out-of-sample)", flush=True)
        print(f"  {'factor':<14} {'bucket':<14} {'n_in':>6} {'WR_in':>7} {'n_oos':>6} {'WR_oos':>7}  hold?", flush=True)
        print(f"  {'-'*100}", flush=True)
        top_cells = []
        for fname, fn in FACTORS.items():
            si_f = si.copy()
            si_f["_f"] = si_f.apply(lambda r: fn(r), axis=1)
            so_f = so.copy()
            so_f["_f"] = so_f.apply(lambda r: fn(r), axis=1)
            for bucket, rows_in in si_f.groupby("_f"):
                w_in, n_in = wr(rows_in)
                if n_in < 100 or w_in is None:
                    continue
                rows_oos = so_f[so_f["_f"] == bucket]
                w_oos, n_oos = wr(rows_oos)
                hold = ""
                if n_oos >= 50 and w_oos is not None:
                    hold = "✓" if w_oos >= max(0.40, w_in - 0.05) else "✗"
                if w_in >= 0.55:
                    top_cells.append((bucket, fname, n_in, w_in, n_oos, w_oos or 0))
                wr_s = f"{w_oos:.1%}" if w_oos is not None else "-"
                print(f"  {fname:<14} {bucket:<14} {n_in:>6} {w_in:>7.1%} {n_oos:>6} {wr_s:>7}  {hold}", flush=True)
        if top_cells:
            print(f"\n  ⭐ IN-SAMPLE CELLS ≥55% WR (n≥100) — do they survive out-of-sample?", flush=True)
            for bucket, fname, n_in, w_in, n_oos, w_oos in sorted(top_cells, key=lambda x: -x[3]):
                verdict = ("SURVIVES" if n_oos >= 50 and w_oos >= max(0.40, w_in - 0.05)
                           else ("too few OOS" if n_oos < 50 else "DID NOT HOLD"))
                print(f"     {strat.upper()} {fname}={bucket:<12} in {w_in:.1%} (n={n_in}) "
                      f"→ oos {w_oos:.1%} (n={n_oos})  [{verdict}]", flush=True)
    print("\n" + "=" * 108, flush=True)
    print("Read: breakeven at 2R is 33% — a cell at 40%+ that HOLDS OOS is a real raw edge.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
