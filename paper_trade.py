#!/usr/bin/env python3
"""paper_trade.py — full-live-parity PAPER trading on the current config.

Replays the exact live pipeline (bot.handle_bar via SimBroker, live GPU veto,
Evolver floor, reflection, PPO exit) over recent REAL bars (CSV + fresh API
bars through Friday's close). No orders ever leave this process.

Usage: .venv/bin/python paper_trade.py [--days 7] [--symbols NQ,ES]
"""
import argparse
import datetime as dt
import sys

import pandas as pd

import config
from sim_broker import SimBroker
from missed_trades import load_bars

def fresh_bars(symbol: str, minutes: int, limit: int = 1000) -> pd.DataFrame:
    """Fetch recent bars from the broker API (read-only — works on any account)."""
    import broker
    c = broker.make_broker(); c.authenticate()
    acct = c.pick_account(config.ACCOUNT)
    contract = c.get_active_contract(symbol)
    df = c.get_bars(contract["id"], minutes, limit=limit)
    return df

def paper_run(symbol: str, days: int):
    import bot
    from backtest import drive, _resolve_specs, WINDOW
    import supervisor
    import evolve
    import reflection as reflection_mem

    config.SYMBOL = symbol
    base = config.base_symbol(symbol)
    tick, tick_value = _resolve_specs(symbol)

    df = load_bars(base)                      # CSV (through Aug 20)
    try:
        fresh = fresh_bars(symbol, config.TIMEFRAME_MIN)
        if not fresh.empty:
            df = pd.concat([df, fresh]).drop_duplicates(subset="time").sort_values("time")
            df = df.reset_index(drop=True)
            print(f"  bars: {len(df)} (CSV + {len(fresh)} fresh API bars, "
                  f"last {df['time'].iloc[-1]})")
    except Exception as e:
        print(f"  ⚠ fresh bars failed ({e}) — CSV only")

    end = df["time"].iloc[-1]
    start = end - pd.Timedelta(days=days)
    start_idx = max(WINDOW, int(df.index[df["time"] >= start][0]) if (df["time"] >= start).any() else WINDOW)

    sim = SimBroker(df, tick, tick_value=tick_value,
                    slip_ticks=config.SLIPPAGE_TICKS,
                    commission_per_side=config.COMMISSION_PER_SIDE_USD)
    ctx = bot.BotContext(sim, account_id=0, contract_id=symbol, tick_size=tick,
                         tick_value=tick_value, log_candles=False)

    # Wire the LIVE GPU veto (fail-closed: sidecar down = no entries) + count blocks
    veto_fn = supervisor.make_veto_fn()
    blocked = {"n": 0}
    def _counting_veto(s, sig, bars):
        ok, why = veto_fn(s, sig, bars)
        if not ok:
            blocked["n"] += 1
        return ok, why
    ctx.veto_fn = _counting_veto

    # Evolver + reflection (supervisor parity): every closed trade -> evolver + lessons
    evolver = evolve.Evolver(baseline_floor=config.PROBA_FLOOR,
                             state_file=os.path.join(os.path.expanduser("~"), ".autotrade_evolve_paper.json"))
    def _record(trade):
        evolver.record(trade)
        reflection_mem.record(trade)
    ctx.on_trade_close = _record
    ctx.evolve_floor = evolver.current_floor()

    names = "+".join(s.name for s in ctx.strategies)
    print(f"▶ PAPER {symbol} [{names}] | conf≥{config.PROBA_FLOOR} ≤{config.PROBA_CEIL} "
          f"| exit: {ctx.exit_mode} | {df['time'].iloc[start_idx]} → {end} "
          f"({len(df) - start_idx} bars)")

    trades = drive(ctx, sim, df, start_idx)
    return trades, blocked["n"], sim

def summarize(trades, symbol, blocked_n):
    if not trades:
        print(f"  {symbol}: 0 trades | {blocked_n} veto blocks")
        return []
    rs = [t.r for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    print(f"  {symbol}: {len(trades)} trades | WR {len(wins)/len(rs):.0%} | "
          f"avgR {sum(rs)/len(rs):+.3f} | sumR {sum(rs):+.2f} | PF {pf:.2f} | "
          f"{blocked_n} veto blocks")
    for t in trades[-6:]:
        side = "LONG" if t.direction > 0 else "SHORT"
        print(f"    {side:5s} {t.strategy:6s} entry {t.entry:.1f} -> exit {t.exit:.1f} "
              f"| {t.r:+.2f}R ({t.reason}, {t.bars_held}b)")
    return rs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--symbols", default="NQ,ES,RTY,YM,GC")
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    all_rs = []
    for sym in symbols:
        try:
            trades, blocked_n, sim = paper_run(sym, args.days)
            rs = summarize(trades, sym, blocked_n)
            all_rs.extend(rs)
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")
    if all_rs:
        print(f"\nBOOK: {len(all_rs)} trades | sumR {sum(all_rs):+.2f} | "
              f"avgR {sum(all_rs)/len(all_rs):+.3f} | "
              f"WR {sum(1 for r in all_rs if r > 0)/len(all_rs):.0%}")

if __name__ == "__main__":
    main()
