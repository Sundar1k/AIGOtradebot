import os
#!/usr/bin/env python3
"""timeframe_compare.py — pure rule-engine backtest across timeframes.

Same mechanical entry on every timeframe (EMA 9/20 cross + ADX gate),
same exit (stop 0.5 x ATR(20), target 2.0R), NO ML grading, NO veto,
no look-ahead. Answers one question with data:

    does a slower candle change the raw edge?

Full history from data/*_3min.csv (2021 -> now), resampled to
3m / 5m / 15m / 60m. Stop-vs-target same-bar: stop wins (conservative).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import indicators as ind

SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
TIMEFRAMES = ["3min", "5min", "15min", "60min"]


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(fos.path.join(os.path.expanduser("~"), "projects/algoTraderBot/data/{symbol}_3min.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("time")
    out = d.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return out


def run(df: pd.DataFrame, symbol: str) -> dict:
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    ef = ind.ema(c, config.EMA_FAST)
    es = ind.ema(c, config.EMA_SLOW)
    atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
    adx = np.asarray(ind.adx(df, config.ADX_P), dtype=float)

    n = len(c)
    warm = max(80, config.ADX_P * 3)
    trades = []
    i = warm
    while i < n - 1:
        if not (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])
                and np.isfinite(adx[i]) and np.isfinite(atr[i])):
            i += 1
            continue
        if config.ADX_GATE and adx[i] < config.ADX_GATE:
            i += 1
            continue
        if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
            d = 1
        elif ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
            d = -1
        else:
            i += 1
            continue

        risk = 0.5 * atr[i]
        if not (risk > 0) or not np.isfinite(risk):
            i += 1
            continue
        entry = c[i]
        stop = entry - d * risk
        tgt = entry + d * 2.0 * risk
        r = None
        j_exit = -1
        j = i + 1
        while j < n:
            if d > 0 and lo[j] <= stop:
                r, j_exit = -1.0, j
                break
            if d < 0 and hi[j] >= stop:
                r, j_exit = -1.0, j
                break
            if d > 0 and hi[j] >= tgt:
                r, j_exit = 2.0, j
                break
            if d < 0 and lo[j] <= tgt:
                r, j_exit = 2.0, j
                break
            j += 1
        if r is None or j_exit < 0:
            break                      # open at end of data — drop
        trades.append((r, j_exit - i))
        i = j_exit + 1

    rs = [t[0] for t in trades]
    n_t = len(rs)
    if n_t == 0:
        return dict(symbol=symbol, n=0, wr=0.0, avg_r=0.0, sum_r=0.0,
                    pf=0.0, days=0.0)
    wins = sum(1 for x in rs if x > 0)
    g_win = sum(x for x in rs if x > 0)
    g_loss = -sum(x for x in rs if x < 0)
    pf = g_win / g_loss if g_loss > 0 else float("inf")
    span_days = (df["time"].iloc[-1] - df["time"].iloc[warm]).total_seconds() / 86400
    return dict(symbol=symbol, n=n_t, wr=round(wins / n_t, 3),
                avg_r=round(sum(rs) / n_t, 3), sum_r=round(sum(rs), 1),
                pf=round(pf, 2), days=round(span_days, 0),
                tpd=round(n_t / max(span_days, 1), 2))


def main():
    print(f"=== PURE RULE-ENGINE TIMEFRAME COMPARISON ===", flush=True)
    print(f"EMA {config.EMA_FAST}/{config.EMA_SLOW} cross + ADX>={config.ADX_GATE} | "
          f"stop 0.5xATR({config.ATR_P}) | target 2R | full history 2021->now | "
          f"no ML, no veto, no look-ahead", flush=True)
    print("=" * 96, flush=True)
    bars = {s: load(s) for s in SYMBOLS}
    agg = {tf: [] for tf in TIMEFRAMES}
    for tf in TIMEFRAMES:
        print(f"\n--- {tf.upper()} candles ---", flush=True)
        for s in SYMBOLS:
            r = run(resample(bars[s], tf), s)
            agg[tf].append(r)
            if r["n"] == 0:
                print(f"  {s}: no trades", flush=True)
            else:
                print(f"  {s}: {r['n']:4d} trades ({r['tpd']}/day) | WR {r['wr']:.1%} | "
                      f"avg {r['avg_r']:+.2f}R | sum {r['sum_r']:+.0f}R | PF {r['pf']:.2f}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print(f"{'TF':>6} {'trades':>7} {'/day':>5} {'WR':>6} {'avgR':>7} {'sumR':>8} {'PF':>6}", flush=True)
    print("-" * 50, flush=True)
    for tf in TIMEFRAMES:
        rs = [r for r in agg[tf] if r["n"] > 0]
        tot_n = sum(r["n"] for r in rs)
        if tot_n == 0:
            print(f"{tf:>6} {'0':>7}", flush=True)
            continue
        tot_r = sum(r["sum_r"] for r in rs)
        wr = sum(r["n"] * r["wr"] for r in rs) / tot_n
        g_win = sum(r["n"] * r["avg_r"] for r in rs if r["avg_r"] > 0)
        g_loss = -sum(r["n"] * r["avg_r"] for r in rs if r["avg_r"] < 0)
        pf = g_win / g_loss if g_loss > 0 else float("inf")
        days = max(r["days"] for r in rs)
        print(f"{tf:>6} {tot_n:>7} {tot_n/days:>5.2f} {wr:>6.1%} "
              f"{tot_r/tot_n:>+7.2f} {tot_r:>+8.0f} {pf:>6.2f}", flush=True)
    print("=" * 96, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
