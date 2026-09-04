#!/usr/bin/env python3
"""bt_adaptive_floor.py — vol-adaptive entry-floor experiment (2026-08-18).

Hypothesis: a STATIC proba floor (0.28) is wrong for both vol regimes —
in high vol, noisy signals deserve a HIGHER floor; in low vol, the
model's confidence is more trustworthy so a LOWER floor is fine.
Test: floor_t = clamp(0.28 + k * (atr_pct_t - 0.5), lo, hi)
where atr_pct_t = percentile rank of ATR(ATR_P) within a trailing
window (default 1000 bars) at bar t.

Compares vs the static 0.28 baseline on the SAME Apr–Jun window the
0.35→0.28 flip was validated on (data ends 2026-06-04), all 5 symbols.

IMPORTANT: this drives the exact production logic (backtest.drive ->
bot.handle_bar) with ctx.evolve_floor set per bar — the same hook the
live Evolver uses. No production code is modified.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config                      # noqa: E402
import bot                         # noqa: E402
from backtest import WINDOW, _load, _resolve_specs   # noqa: E402
from sim_broker import SimBroker   # noqa: E402
from indicators import atr as atr_fn   # noqa: E402


# ── adaptive floor formula ─────────────────────────────────────────────
def make_floor_fn(k: float, lo: float, hi: float, lookback: int):
    """Return fn(atr_pct) -> floor, clamped to [lo, hi]."""
    def floor_fn(atr_pct: float) -> float:
        f = config.PROBA_FLOOR + k * (atr_pct - 0.5)
        return float(np.clip(f, lo, hi))
    return floor_fn


def atr_percentile_series(df: pd.DataFrame, period: int, lookback: int) -> np.ndarray:
    """Percentile rank of ATR(period) at each bar within the trailing
    `lookback` window (rank of the LAST value in each window). Causal:
    bar t uses only bars [t-lookback+1, t]."""
    a = atr_fn(df, period)                       # (n,) ATR series
    s = pd.Series(a)
    pct = s.rolling(lookback, min_periods=min(lookback // 4, 200)).rank(pct=True)
    return pct.to_numpy()


def drive_adaptive(ctx, sim, df, start_idx, floor_fn, atr_pct, window=WINDOW):
    """Same as backtest.drive but sets ctx.evolve_floor per bar from the
    adaptive floor fn. Returns (trades, floor_used_list)."""
    trade_state = None
    floors = []
    for i in range(start_idx, len(df)):
        sim.set_bar(i)
        sim.process_exits()
        if sim.pos is None:
            trade_state = None
        f = floor_fn(atr_pct[i])
        ctx.evolve_floor = f
        floors.append(f)
        win = df.iloc[max(0, i - window + 1): i + 1]
        trade_state = bot.handle_bar(ctx, win, trade_state)
        if sim.pos is not None and trade_state and sim.pos.get("strategy") is None:
            sim.tag_strategy(trade_state["strategy"].name)
    sim.close_open()
    return sim.trades, floors


def _stats(trades):
    if not trades:
        return {"n": 0, "wr": 0.0, "mean_r": 0.0, "sum_r": 0.0, "pf": 0.0}
    r = np.array([t.r for t in trades])
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return {"n": len(r), "wr": 100 * (r > 0).mean(), "mean_r": r.mean(),
            "sum_r": r.sum(), "pf": pf}


def run_symbol(symbol, start, end, variants, lookback):
    """Run all variants for one symbol; returns rows for the table."""
    config.SYMBOL = symbol
    base = config.base_symbol(symbol)
    tick, tick_value = _resolve_specs(symbol)     # broker API (creds in .env)
    df = _load(base, end)
    ts = pd.Timestamp(start, tz="UTC")
    hits = df.index[df["time"] >= ts]
    start_idx = max(WINDOW, int(hits[0]) if len(hits) else len(df))

    atr_pct = atr_percentile_series(df, config.ATR_P, lookback)

    rows = []
    # baseline: static floor (evolve_floor=None -> config.PROBA_FLOOR)
    sim = SimBroker(df, tick)
    ctx = bot.BotContext(sim, account_id=0, contract_id=symbol,
                         tick_size=tick, tick_value=tick_value,
                         log_candles=False)
    t0 = time.time()
    trades, _ = drive_adaptive(ctx, sim, df, start_idx, lambda p: config.PROBA_FLOOR, atr_pct)
    st = _stats(trades)
    rows.append(("static-0.28", st, time.time() - t0))

    for (k, lo, hi) in variants:
        sim = SimBroker(df, tick)
        ctx = bot.BotContext(sim, account_id=0, contract_id=symbol,
                             tick_size=tick, tick_value=tick_value,
                             log_candles=False)
        fn = make_floor_fn(k, lo, hi, lookback)
        t0 = time.time()
        trades, floors = drive_adaptive(ctx, sim, df, start_idx, fn, atr_pct)
        st = _stats(trades)
        st["floor_lo"] = float(np.min(floors)) if floors else 0.0
        st["floor_hi"] = float(np.max(floors)) if floors else 0.0
        rows.append((f"adapt k={k} [{lo},{hi}]", st, time.time() - t0))
    return symbol, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-06-05")
    ap.add_argument("--symbols", default="NQ,ES,RTY,YM,GC")
    ap.add_argument("--lookback", type=int, default=1000)
    ap.add_argument("--variants", default="0.10:0.24:0.50,0.15:0.24:0.50,0.20:0.28:0.50")
    args = ap.parse_args()

    variants = []
    for v in args.variants.split(","):
        k, lo, hi = (float(x) for x in v.split(":"))
        variants.append((k, lo, hi))

    print(f"adaptive-floor experiment | window {args.start}→{args.end} "
          f"| lookback {args.lookback} | variants {variants}")
    print("=" * 100)
    hdr = (f"{'symbol':6s} {'variant':24s} {'n':>5s} {'WR%':>6s} "
           f"{'meanR':>7s} {'sumR':>8s} {'PF':>6s} {'floor range':>14s} {'secs':>6s}")
    print(hdr)
    print("-" * 100)
    for symbol in args.symbols.split(","):
        symbol = symbol.strip().upper()
        try:
            sym, rows = run_symbol(symbol, args.start, args.end, variants, args.lookback)
        except Exception as e:
            print(f"{symbol:6s} ERROR: {e}")
            continue
        for name, st, secs in rows:
            fr = ""
            if "floor_lo" in st:
                fr = f"[{st['floor_lo']:.2f},{st['floor_hi']:.2f}]"
            print(f"{sym:6s} {name:24s} {st['n']:5d} {st['wr']:6.1f} "
                  f"{st['mean_r']:+7.3f} {st['sum_r']:+8.2f} {st['pf']:6.2f} "
                  f"{fr:>14s} {secs:6.0f}", flush=True)
    print("=" * 100)


if __name__ == "__main__":
    main()
