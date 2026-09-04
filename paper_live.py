#!/usr/bin/env python3
"""paper_live.py — live-PAPER trading daemon for CME futures (TopstepX data).

The exact live pipeline (bot.handle_bar + SimBroker fills + live GPU veto +
Evolver floor + reflection + PPO exit) driven by REAL bars fetched from the
broker API every bar close. NO ORDERS ARE EVER SENT — everything fills against
SimBroker. Run it whenever you want a live-market paper book of the current
config (ACTIVE_STRATEGIES, floor/ceil, PPO exit).

State: ~/.autotrade_paper_state.json (heartbeat, halted, paper balance, trades)
Logs:  journalctl --user -u paper-trade.service
Modes: --once = single cycle then exit (smoke test)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

import config
from sim_broker import SimBroker

STATE = Path.home() / ".autotrade_paper_state.json"
WINDOW = 160                     # bars of context handle_bar sees (>= CTX+30)

# paper breakers (same shape as the live supervisor, paper-only)
DAILY_LOSS_LIMIT = float(__import__("os").environ.get("AUTOTRADE_DAILY_LOSS", "400"))
PROFIT_TARGET = float(__import__("os").environ.get("AUTOTRADE_PROFIT_TARGET", "500"))
PROFIT_BAND = float(__import__("os").environ.get("AUTOTRADE_PROFIT_BAND", "150"))


def log(msg: str):
    print(msg, flush=True)


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"halted": False, "reason": "", "start_balance": None,
                "at_peak": None, "trades": [], "last_bar": {}, "date": ""}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str))
    tmp.replace(STATE)


def fetch_live(client, contract_id: str, minutes: int, limit: int = 1000) -> pd.DataFrame:
    df = client.get_bars(contract_id, minutes, limit=limit)
    if df.empty:
        return df
    return df.sort_values("time").reset_index(drop=True)


class PaperBook:
    """One paper account across all symbols (one position max, like live)."""

    def __init__(self):
        self.sims = {}            # symbol -> SimBroker
        self.ctxs = {}            # symbol -> BotContext
        self.trade_state = {}     # symbol -> handle_bar trade_state
        self.blocked = {}         # symbol -> veto-block count
        self.start_balance = 100_000.0     # paper starting equity
        self.equity = self.start_balance

    def any_open(self):
        return any(s.pos is not None for s in self.sims.values())


def build_book(symbols, evolver, veto_fn):
    """Fetch initial bars + contracts, build sims/contexts for every symbol."""
    import broker
    import bot
    import supervisor
    from strategies import make_strategies as _mk

    client = broker.make_broker()
    client.authenticate()
    acct = client.pick_account(config.ACCOUNT)

    book = PaperBook()
    for sym in symbols:
        config.SYMBOL = sym
        contract = client.get_active_contract(sym)
        tick = float(contract["tickSize"])
        tick_value = float(contract["tickValue"])
        df = fetch_live(client, contract["id"], config.TIMEFRAME_MIN)
        if len(df) < WINDOW + 40:
            # cold-start padding: merge local CSV history with the fresh bars so
            # the book can build early in the trading week (the broker only
            # returns ~170 3-min bars right after the Sunday reopen).
            from missed_trades import load_bars
            df = load_bars(sym)
            log(f"  📥 {sym}: {len(df)} bars (CSV+API padded)")
        if len(df) < WINDOW + 40:
            log(f"  ⚠ {sym}: only {len(df)} bars — skipping (need {WINDOW + 40})")
            continue
        sim = SimBroker(df, tick, tick_value=tick_value,
                        slip_ticks=config.SLIPPAGE_TICKS,
                        commission_per_side=config.COMMISSION_PER_SIDE_USD)
        # live parity: orb is auto-added for ORB symbols (supervisor L391-394)
        _active = list(config.ACTIVE_STRATEGIES)
        if sym in config.ORB_SYMBOLS and "orb" not in _active:
            _active.append("orb")
        ctx = bot.BotContext(sim, account_id=acct["id"], contract_id=contract["id"],
                             tick_size=tick, tick_value=tick_value,
                             strategies=_mk(_active), log_candles=False)
        ctx.veto_fn = veto_fn
        ctx.entry_gate = supervisor.entry_gate   # 09:30-12:00 ET (cycle-3 GO)
        ctx.on_trade_close = evolver.record
        ctx.evolve_floor = evolver.current_floor()
        ctx.symbol = sym
        book.sims[sym] = sim
        book.ctxs[sym] = ctx
        book.trade_state[sym] = None
        book.blocked[sym] = 0
        log(f"  🛰 {sym} -> {contract['id']} (tick {tick:g}) "
            f"[{'+'.join(s.name for s in ctx.strategies)}] {len(df)} bars")
    return client, book


def one_cycle(client, book, evolver, symbols, st):
    """Process every NEW bar for every symbol since the last cycle."""
    import bot
    import supervisor

    new_any = False
    for sym in symbols:
        if sym not in book.sims:
            continue
        sim, ctx = book.sims[sym], book.ctxs[sym]
        config.SYMBOL = sym
        contract = client.get_active_contract(sym)
        fresh = fetch_live(client, contract["id"], config.TIMEFRAME_MIN)
        if fresh.empty:
            continue
        last_seen = st["last_bar"].get(sym)
        new_rows = fresh[fresh["time"] > last_seen] if last_seen else fresh.iloc[WINDOW:]
        if new_rows.empty:
            continue
        new_any = True
        n0 = len(sim.trades)
        for _, row in new_rows.iterrows():
            sim.df.loc[len(sim.df)] = row          # grow the sim's frame in place
            i = len(sim.df) - 1
            sim.set_bar(i)
            sim.process_exits()                   # resting stop/target fills first
            if sim.pos is None:
                book.trade_state[sym] = None
            win = sim.df.iloc[max(0, i - WINDOW + 1): i + 1]
            # one-position-max across the paper book: when another symbol holds
            # the position, flat symbols skip entries this bar (live parity)
            if book.any_open() and sim.pos is None and book.trade_state[sym] is None:
                continue
            book.trade_state[sym] = bot.handle_bar(ctx, win, book.trade_state[sym])
            if sim.pos is not None and sim.pos.get("strategy") is None:
                src = book.trade_state[sym] or getattr(ctx, "last_trade", None)
                if src and src.get("strategy") is not None:
                    sim.tag_strategy(src["strategy"].name)
            ctx.evolve_floor = evolver.current_floor()
            st["last_bar"][sym] = str(row["time"])
        # settle P&L from the trades this cycle produced
        for t in sim.trades[n0:]:
            book.equity += t.r * t.risk * ctx.tick_value
        if len(sim.trades) > n0:
            st["trades"] = [{"sym": sym, "r": t.r, "strategy": t.strategy,
                             "reason": t.reason, "entry": t.entry, "exit": t.exit,
                             "when": str(t.entry_time)} for t in sim.trades[-50:]]
    return new_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--symbols", default="NQ,ES,RTY,YM,GC")
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]

    import supervisor
    import evolve
    import reflection as reflection_mem

    evolver = evolve.Evolver(baseline_floor=config.PROBA_FLOOR,
                             state_file=os.path.join(os.path.expanduser("~"), ".autotrade_evolve_paper.json"))
    _orig = evolver.record
    def _record(trade):
        _orig(trade)
        reflection_mem.record(trade)
    evolver.record = _record

    st = load_state()
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    if st.get("date") != today:
        st.update({"date": today, "halted": False, "reason": "",
                   "start_balance": None, "at_peak": None})
    st.setdefault("last_bar", {})
    st["start_balance"] = st.get("start_balance") or 100_000.0

    log(f"=== PAPER LIVE 2026-08-22 | symbols={','.join(symbols)} | "
        f"conf≥{config.PROBA_FLOOR} ≤{config.PROBA_CEIL} | "
        f"entry-window: {'09:30-12:00 ET [ON]' if supervisor.ENTRY_WINDOW else '24h [OFF]'} | "
        f"veto {supervisor.VETO_MODE} | daily-loss ${DAILY_LOSS_LIMIT:.0f} ===")

    client, book = build_book(symbols, evolver, supervisor.make_veto_fn())
    if not book.sims:
        log("✗ no symbols built — abort")
        return 1
    # sync last-seen bar to the book's newest bar so the first cycle doesn't
    # re-append history the book already holds (matters after CSV padding).
    for sym in book.sims:
        st["last_bar"][sym] = str(book.sims[sym].df["time"].iloc[-1])
    save_state(st)
    log(f"✅ paper book ready: {list(book.sims)} | {len(next(iter(book.sims.values())).df)} "
        f"bars each | waiting for new bars…")

    while True:
        now = dt.datetime.now(dt.timezone.utc)
        if supervisor.in_rth(now):
            st["halted"] = st.get("halted", False)
            if not st["halted"]:
                try:
                    new_any = one_cycle(client, book, evolver, symbols, st)
                    if new_any:
                        log(f"  [{now.isoformat(timespec='minutes')}] paper equity "
                            f"${book.equity:,.2f} | trades {len(st['trades'])}")
                except Exception as e:
                    log(f"  ⚠ cycle error: {e}")
        # paper breakers
        if not st.get("halted"):
            day_pnl = book.equity - (st.get("start_balance") or book.equity)
            if day_pnl <= -DAILY_LOSS_LIMIT:
                st.update({"halted": True, "reason": f"paper daily loss ${day_pnl:,.0f} "
                          f"<= -${DAILY_LOSS_LIMIT:.0f}"})
                log(f"⛔ PAPER HALTED: {st['reason']}")
            elif book.equity >= (st.get("start_balance") or book.equity) + PROFIT_TARGET - PROFIT_BAND:
                st["reason"] = f"paper profit target zone (${book.equity:,.0f})"
        st["last_beat"] = now.isoformat()
        save_state(st)
        if args.once:
            log("--once: single cycle done")
            return 0
        period = config.TIMEFRAME_MIN * 60
        time.sleep(period - (time.time() % period) + 2)


if __name__ == "__main__":
    sys.exit(main())
