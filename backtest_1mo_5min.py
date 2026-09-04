#!/usr/bin/env python3
"""backtest_1mo_5min.py — 1-month backtest on 5-MINUTE candles.

⚠️ INDICATIVE-ONLY: the models were TRAINED on 3-min bars. There is no
5-min model file (the code normally forbids cross-timeframe use). This runs
the 3-min model on 5-min bars to see how the ENGINE behaves at a slower
resolution — results are directional insight, NOT live-trustworthy.

Pipeline: same as live (EMA(9/20)+ADX≥18 → grade → floor 0.35/ceil 0.50 →
2R bracket), bars resampled 3-min → 5-min from CSV+broker data.

Usage: ./.venv/bin/python backtest_1mo_5min.py [--days 30]
"""
import argparse
import datetime as dt
import sys

import config
import pandas as pd
from sim_broker import SimBroker
from missed_trades import load_bars

RESAMPLE = "5min"


def resample(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().set_index("time")
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    out = d.resample(RESAMPLE).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return out


def run_symbol(symbol: str, days: int) -> dict:
    import bot
    df3 = load_bars(symbol)
    if len(df3) < 600:
        return {"symbol": symbol, "error": f"insufficient bars ({len(df3)})"}
    df = resample(df3)
    cut = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    df = df[df["time"] >= cut].reset_index(drop=True)
    if len(df) < 300:
        return {"symbol": symbol, "error": f"insufficient 5m bars in window ({len(df)})"}

    config.SYMBOL = symbol
    try:
        import broker
        tick, tv = broker.fetch_contract_specs(symbol)
    except Exception:
        tick = 0.25 if symbol in ("NQ", "ES") else 0.1
        tv = 0.0

    sim = SimBroker(df, tick)
    ctx = bot.BotContext(sim, 0, symbol, tick, tv, log_candles=False)
    ctx.symbol = symbol

    from backtest import drive
    trades = drive(ctx, sim, df, 400)

    n = len(trades)
    if n == 0:
        return {"symbol": symbol, "n": 0, "wr": 0.0, "avg_r": 0.0,
                "sum_r": 0.0, "pf": 0.0,
                "window": f"{str(df['time'].iloc[400])[:10]} → {str(df['time'].iloc[-1])[:10]}"}
    rs = [t.r for t in trades]
    wins = sum(1 for r in rs if r > 0)
    g_win = sum(r for r in rs if r > 0)
    g_loss = -sum(r for r in rs if r < 0)
    pf = g_win / g_loss if g_loss > 0 else float("inf")
    by_exit = {}
    for t in trades:
        by_exit.setdefault(getattr(t, "reason", "?"), []).append(t.r)
    return {"symbol": symbol, "n": n, "wr": round(wins / n, 3),
            "avg_r": round(sum(rs) / n, 3), "sum_r": round(sum(rs), 2),
            "pf": round(pf, 2),
            "exits": "; ".join(f"{k}:{len(v)}" for k, v in by_exit.items()),
            "window": f"{str(df['time'].iloc[400])[:10]} → {str(df['time'].iloc[-1])[:10]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    print(f"=== 1-MONTH BACKTEST — {args.days} days on 5-MIN candles ===", flush=True)
    print("⚠️ INDICATIVE ONLY — models trained on 3-min; no 5-min model exists.", flush=True)
    print("=" * 72, flush=True)

    results = []
    for sym in config.TRADE_SYMBOLS:
        r = run_symbol(sym, args.days)
        results.append(r)
        if "error" in r:
            print(f"  ❌ {sym}: {r['error']}", flush=True)
            continue
        print(f"  {sym}: {r['n']:2d} trades | WR {r['wr']:.1%} | "
              f"avg {r['avg_r']:+.2f}R | sum {r['sum_r']:+.1f}R | PF {r['pf']:.2f} | "
              f"exits[{r.get('exits', '?')}]", flush=True)
        print(f"       window {r['window']}", flush=True)

    ok = [r for r in results if "error" not in r]
    tot_n = sum(r["n"] for r in ok)
    if tot_n:
        tot_r = sum(r["sum_r"] for r in ok)
        wins = sum(round(r["n"] * r["wr"]) for r in ok)
        g_win = sum(r["n"] * r["avg_r"] for r in ok if r["avg_r"] > 0)
        g_loss = -sum(r["n"] * r["avg_r"] for r in ok if r["avg_r"] < 0)
        pf = g_win / g_loss if g_loss > 0 else float("inf")
        print("=" * 72)
        print(f"TOTAL (5-min, indicative): {tot_n} trades | WR {wins/tot_n:.1%} | "
              f"sum {tot_r:+.1f}R | PF {pf:.2f} | avg {tot_r/tot_n:+.2f}R/trade")
        print("=" * 72)
        print("Compare with the 3-min 2-week result: 16 trades, +1.1R, PF 1.34.")
        print("More trades at 5-min = slower resolution, fewer signals; the")
        print("model is out-of-distribution — trust the 3-min numbers for live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
