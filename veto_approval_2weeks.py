#!/usr/bin/env python3
"""veto_approval_2weeks.py — replay the 2-week backtest and ask the LIVE GPU
veto whether it would have approved each engine trade.

Same pipeline as backtest_2weeks.py (exact live handle_bar via SimBroker),
but at each ENTRY it captures the exact veto state line (build_state_line —
byte-identical to what the live supervisor POSTs) and batch-calls
:8765/decide_batch. Output: per trade, engine side vs veto action, and the
live-filtered P&L (engine trades the veto APPROVES only).

Usage: ./.venv/bin/python veto_approval_2weeks.py [--days 14]
"""
import argparse
import datetime as dt
import sys

import requests

import config
from sim_broker import SimBroker
from missed_trades import load_bars

VETO_URL = "http://127.0.0.1:8765/decide_batch"


def replay_with_states(symbol: str, days: int):
    """Run the exact backtest; yield (trade, state_text) per entry."""
    import bot
    from backtest import drive

    df = load_bars(symbol)
    if len(df) < 500:
        return
    cut = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    df = df[df["time"] >= cut].reset_index(drop=True)
    if len(df) < 400:
        return

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

    # hook: capture the state line at each entry
    from supervisor import build_state_line
    captured = []

    # simplest: wrap handle_bar at module level
    _orig_handle = bot.handle_bar

    def _handle(c, bars, trade_state):
        prev_pos = sim.pos is not None
        ts = _orig_handle(c, bars, trade_state)
        if sim.pos is not None and not prev_pos:
            try:
                # entry_idx is the bar index in df; trades carry entry_time —
                # convert idx -> df time -> match trades
                idx = sim.pos.get("entry_idx")
                et = str(df["time"].iloc[idx]) if idx is not None else None
                captured.append((et, build_state_line(bars, c.symbol)))
            except Exception as e:
                print(f"  ⚠️ state capture failed: {e}", flush=True)
        return ts

    bot.handle_bar = _handle
    try:
        trades = drive(ctx, sim, df, 400)
    finally:
        bot.handle_bar = _orig_handle

    by_entry = {str(t.entry_time): t for t in trades}
    for et, state in captured:
        t = by_entry.get(et)
        if t is not None:
            yield t, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    print(f"=== VETO APPROVAL REPLAY — last {args.days} days ===", flush=True)
    print("Engine: EMA+XGB floor/ceil 2R bracket (as backtest). Veto: LIVE GPU "
          "7B model, cache-first.", flush=True)
    print("=" * 72, flush=True)

    all_trades = []       # (symbol, trade, state, veto_action)
    for sym in config.TRADE_SYMBOLS:
        print(f"  replaying {sym}...", flush=True)
        for t, state in replay_with_states(sym, args.days):
            all_trades.append((sym, t, state))
            print(f"    entry {t.entry_time} {sym} "
                  f"{'LONG' if t.direction > 0 else 'SHORT'} "
                  f"R={t.r:+.2f} ({t.reason})", flush=True)

    if not all_trades:
        print("no engine trades in window", flush=True)
        return

    # batch-call the veto
    print(f"\n  asking live veto on {len(all_trades)} states...", flush=True)
    texts = [s for _, _, s in all_trades]
    resp = requests.post(VETO_URL, json={"texts": texts}, timeout=600)
    resp.raise_for_status()
    results = resp.json()["results"]
    for i, (sym, t, state) in enumerate(all_trades):
        action = results[i].get("action", "NO TRADE")
        all_trades[i] = (sym, t, state, action)

    # verdict per trade
    print("=" * 72)
    print(f"{'SYM':4s} {'SIDE':5s} {'VETO':9s} {'APPROVE':7s} {'R':>6s}  exit")
    approved = []
    for sym, t, state, action in all_trades:
        want = "BUY" if t.direction > 0 else "SELL"
        ok = action == want
        flag = "✅" if ok else "⛔"
        if ok:
            approved.append(t)
        print(f"{sym:4s} {want:5s} {action:9s} {flag:7s} {t.r:+.2f}  {t.reason}")

    eng_n, eng_r = len(all_trades), sum(t.r for _, t, _, _ in all_trades)
    vet_n, vet_r = len(approved), sum(t.r for t in approved)
    eng_wr = sum(1 for _, t, _, _ in all_trades if t.r > 0) / eng_n
    vet_wr = sum(1 for t in approved if t.r > 0) / vet_n if vet_n else 0.0

    print("=" * 72)
    print(f"ENGINE alone:  {eng_n} trades | WR {eng_wr:.1%} | sum {eng_r:+.1f}R")
    print(f"VETO-APPROVED: {vet_n} trades | WR {vet_wr:.1%} | sum {vet_r:+.1f}R "
          f"| blocked {eng_n - vet_n}")
    print("=" * 72)
    if vet_n:
        print(f"live-filtered expectancy: {vet_r/vet_n:+.2f}R/trade "
              f"(vs engine {eng_r/eng_n:+.2f}R/trade)")
    else:
        print("veto approved NOTHING — engine alone would never have traded live")


if __name__ == "__main__":
    sys.exit(main())
