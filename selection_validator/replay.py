"""replay.py — selection-validator replay runner (spec Phase 1).

Drives the EXACT live per-bar logic (bot.handle_bar) through SimBroker over
local CSVs for every traded symbol, recording every graded signal
point-in-time (take AND skip) with realized outcomes for taken+closed trades.

Per-symbol strategy roster mirrors the live supervisor:
  ACTIVE_STRATEGIES + orb where config.ORB_SYMBOLS says it wins.
Cost model mirrors live: config.SLIPPAGE_TICKS + COMMISSION_PER_SIDE_USD.

Usage:  python -m selection_validator.replay [--symbols NQ,ES,RTY,YM,GC] [--end YYYY-MM-DD]
Output: selection_validator/data/signals_<SYM>.jsonl (append-safe per symbol)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import config                                # noqa: E402
import bot                                   # noqa: E402
from backtest import _load, _resolve_specs, drive  # noqa: E402
from selection_validator.dataset import SignalSink  # noqa: E402
from sim_broker import SimBroker             # noqa: E402
from strategies import make_strategies       # noqa: E402

OUT_DIR = os.path.join(BASE, "selection_validator", "data")


def build_ctx(symbol: str):
    """BotContext with the live supervisor's per-symbol strategy roster."""
    tick, tick_value = _resolve_specs(symbol)
    df = _load(config.base_symbol(symbol), None)
    sim = SimBroker(df, tick, tick_value=tick_value,
                    slip_ticks=config.SLIPPAGE_TICKS,
                    commission_per_side=config.COMMISSION_PER_SIDE_USD)
    active = list(config.ACTIVE_STRATEGIES)
    if symbol in config.ORB_SYMBOLS and "orb" not in active:
        active.append("orb")
    ctx = bot.BotContext(sim, account_id=0, contract_id=symbol, tick_size=tick,
                         tick_value=tick_value, log_candles=False,
                         strategies=make_strategies(active))
    ctx.symbol = symbol
    return ctx, sim, df


def replay_symbol(symbol: str, end: str | None, progress_every: int = 25_000):
    t0 = time.time()
    ctx, sim, df = build_ctx(symbol)
    if end:
        df = df[df["time"] < pd.Timestamp(end, tz="UTC")].reset_index(drop=True)
    sink = SignalSink(os.path.join(OUT_DIR, f"signals_{symbol}.jsonl"), symbol)
    n = len(df)

    def _probe(i):
        if i % progress_every == 0:
            el = time.time() - t0
            rate = (i - 500) / max(el, 1e-9)
            eta = (n - i) / max(rate, 1e-9) / 60
            print(f"  {symbol}: bar {i}/{n} ({100*i/n:.1f}%) | {rate:.0f} bars/s | "
                  f"ETA {eta:.0f} min", flush=True)

    _probe(500)
    trades = drive(ctx, sim, df, 500, signal_sink=sink, window=500)
    sink.close()
    from selection_validator.dataset import merge_outcomes
    matched = merge_outcomes(os.path.join(OUT_DIR, f"signals_{symbol}.jsonl"), trades)
    el = time.time() - t0
    print(f"✅ {symbol}: {len(df)-500} bars in {el/60:.1f} min | "
          f"{len(trades)} trades ({matched} outcomes matched) | "
          f"signals -> signals_{symbol}.jsonl", flush=True)
    return len(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(config.TRADE_SYMBOLS))
    ap.add_argument("--end", default=None, help="replay only bars < this date (UTC)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        if config.base_symbol(sym) not in config.TRAINED_SYMBOLS:
            print(f"skip {sym}: not in TRAINED_SYMBOLS", flush=True)
            continue
        total += replay_symbol(sym, args.end)
    print(f"=== replay complete: {total} trades across symbols ===", flush=True)


if __name__ == "__main__":
    main()
