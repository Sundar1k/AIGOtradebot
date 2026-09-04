#!/usr/bin/env python3
"""backtest_2weeks.py — full-framework backtest of the LAST 14 DAYS.

Drives the EXACT live per-bar logic (bot.handle_bar via SimBroker — same as
backtest.py's drive()) over merged data: CSV history + fresh broker bars, so
the full 2-week window is covered. For every symbol: detect → grade →
floor/ceil → 2R bracket, exactly as live (minus the GPU veto, which is a
live-only filter — noted in output).

Usage: ./.venv/bin/python backtest_2weeks.py [--days 14]
"""
import argparse
import datetime as dt
import sys

import config
import pandas as pd
from sim_broker import SimBroker
from missed_trades import load_bars


def run_symbol(symbol: str, days: int) -> dict:
    import bot
    df = load_bars(symbol)
    if len(df) < 500:
        return {"symbol": symbol, "error": f"insufficient bars ({len(df)})"}
    # crop to the window
    cut = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    df = df[df["time"] >= cut].reset_index(drop=True)
    if len(df) < 400:
        return {"symbol": symbol, "error": f"insufficient bars in window ({len(df)})"}

    config.SYMBOL = symbol
    base = config.base_symbol(symbol)
    tick, tick_value = 0.0, 0.0
    try:
        import broker
        ts, tv = broker.fetch_contract_specs(symbol)
        tick, tick_value = ts, tv
    except Exception as e:
        print(f"  ⚠️ {symbol}: specs fetch failed ({e}) — using defaults", flush=True)
        tick = 0.25 if "NQ" in base or "ES" in base else 0.1

    sim = SimBroker(df, tick)
    ctx = bot.BotContext(sim, account_id=0, contract_id=symbol, tick_size=tick,
                         tick_value=tick_value, log_candles=False)
    ctx.symbol = symbol
    start_idx = 400  # indicator + embed warmup

    from backtest import drive
    trades = drive(ctx, sim, df, start_idx)

    n = len(trades)
    if n == 0:
        return {"symbol": symbol, "n": 0, "wr": 0.0, "avg_r": 0.0,
                "sum_r": 0.0, "pf": 0.0, "window_start": str(df["time"].iloc[start_idx])[:16],
                "window_end": str(df["time"].iloc[-1])[:16]}
    rs = [t.r for t in trades]
    wins = sum(1 for r in rs if r > 0)
    gross_win = sum(r for r in rs if r > 0)
    gross_loss = -sum(r for r in rs if r < 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    by_exit = {}
    for t in trades:
        by_exit.setdefault(getattr(t, "reason", "?"), []).append(t.r)
    exit_str = "; ".join(f"{k}:{len(v)}" for k, v in by_exit.items())
    return {"symbol": symbol, "n": n, "wr": round(wins / n, 3),
            "avg_r": round(sum(rs) / n, 3), "sum_r": round(sum(rs), 2),
            "pf": round(pf, 2), "exits": exit_str,
            "window_start": str(df["time"].iloc[start_idx])[:16],
            "window_end": str(df["time"].iloc[-1])[:16]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    print(f"=== FULL-FRAMEWORK BACKTEST — last {args.days} days ===", flush=True)
    print("Pipeline: EMA(9/20)+ADX≥18 → Chronos+XGB grade → floor 0.35/ceil 0.50 "
          "→ fixed 2R bracket", flush=True)
    print("Live-only filters NOT simulated: GPU veto, news blackout, regime gate, "
          "conflict gate, breakers", flush=True)
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
        print(f"       window {r['window_start']} → {r['window_end']}", flush=True)

    # aggregate
    ok = [r for r in results if "error" not in r]
    tot_n = sum(r["n"] for r in ok)
    if tot_n:
        tot_r = sum(r["sum_r"] for r in ok)
        wins = sum(round(r["n"] * r["wr"]) for r in ok)
        g_win = sum(r["n"] * r["avg_r"] for r in ok if r["avg_r"] > 0)
        g_loss = -sum(r["n"] * r["avg_r"] for r in ok if r["avg_r"] < 0)
        pf = g_win / g_loss if g_loss > 0 else float("inf")
        print("=" * 72)
        print(f"TOTAL: {tot_n} trades | win rate {wins/tot_n:.1%} | "
              f"sum {tot_r:+.1f}R | PF {pf:.2f} | avg {tot_r/tot_n:+.2f}R/trade")
        print("=" * 72)
        print("\nLive veto NOT in backtest — the 7B LLM is a filter ON TOP of "
              "this engine. Expect live trades ⊆ backtest signals, and veto "
              "agreement historically ≈ 60-70% of engine signals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
