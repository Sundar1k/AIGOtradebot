import os
#!/usr/bin/env python3
"""audit_vol_jump.py — two point-in-time experiments on the engine (no veto):

  1. JUMP FILTER A/B: does skipping signals on/after jump bars help or hurt?
     (config.JUMP_ATR_MULT 0 vs 3.0)
  2. VOL-GATE DIRECTION: bucket engine trades by TRAILING ATR percentile at
     entry — are high-vol entries better or worse? (answers whether the regime
     gate's "block panic/high-vol" direction is right or inverted)

Same blind OOS window (W = start -> data end), same live pipeline
(bot.handle_bar + SimBroker + PPO exit + slippage), no look-ahead.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import bot
import indicators as ind
import backtest
from backtest import drive, _resolve_specs, WINDOW
from sim_broker import SimBroker

SYM = sys.argv[1] if len(sys.argv) > 1 else "NQ"
START = sys.argv[2] if len(sys.argv) > 2 else "2026-04-21"


def stats(trades, label):
    r = np.array([t.r for t in trades]) if trades else np.array([])
    if len(r) == 0:
        print(f"{label}: 0 trades")
        return
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    print(f"{label}: n={len(r)}  WR={100*(r > 0).mean():.1f}%  "
          f"meanR={r.mean():+.3f}  sumR={r.sum():+.2f}  PF={pf:.2f}")


def run(jump_mult):
    config.JUMP_ATR_MULT = jump_mult
    sim = SimBroker(df, tick, tick_value=tick_value,
                    slip_ticks=config.SLIPPAGE_TICKS,
                    commission_per_side=config.COMMISSION_PER_SIDE_USD)
    ctx = bot.BotContext(sim, account_id=0, contract_id=SYM, tick_size=tick,
                         tick_value=tick_value, log_candles=False)
    return drive(ctx, sim, df, start_idx)


config.SYMBOL = SYM
base = config.base_symbol(SYM)
tick, tick_value = _resolve_specs(SYM)
df = backtest._load(base, None)
start_idx = max(WINDOW, int(df.index[df["time"] >= pd.Timestamp(START, tz="UTC")][0]))
print(f"=== {SYM} | {START} -> {df['time'].iloc[-1].date()} | "
      f"{len(df) - start_idx} bars | slip {config.SLIPPAGE_TICKS}t/side ===\n")

print("--- 1. JUMP FILTER A/B ---")
base_trades = run(0.0)      # jump filter OFF
jump_trades = run(3.0)      # jump filter ON (3.0 x ATR)
stats(base_trades, "jump OFF")
stats(jump_trades, "jump ON ")

print("\n--- 2. VOL-GATE DIRECTION (trailing ATR percentile at entry) ---")
atr = np.asarray(ind.atr(df, config.ATR_P), dtype=float)
time_idx = {ts: i for i, ts in enumerate(df["time"])}


def trailing_pctile(idx, lookback=500):
    lo = max(0, idx - lookback + 1)
    win = atr[lo:idx + 1]
    win = win[np.isfinite(win) & (win > 0)]
    if len(win) < 50 or not np.isfinite(atr[idx]) or atr[idx] <= 0:
        return None
    return float((win <= atr[idx]).mean())


rows = []
for t in base_trades:
    idx = time_idx.get(t.entry_time)
    if idx is None:
        continue
    pct = trailing_pctile(idx)
    if pct is None:
        continue
    rows.append((pct, t.r))

if rows:
    pcts = np.array([x[0] for x in rows])
    rs = np.array([x[1] for x in rows])
    edges = [0.0, 0.25, 0.50, 0.75, 1.0]
    print(f"{'vol bucket':>12} {'n':>4} {'WR':>6} {'meanR':>8} {'sumR':>8} {'PF':>6}")
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (pcts >= lo) & (pcts <= hi)
        if not m.any():
            continue
        rr = rs[m]
        w = rr[rr > 0].sum()
        l = -rr[rr < 0].sum()
        pf = w / l if l > 0 else float("inf")
        print(f"{f'{lo*100:.0f}-{hi*100:.0f}%':>12} {m.sum():>4} "
              f"{100*(rr > 0).mean():>5.1f}% {rr.mean():>+8.3f} "
              f"{rr.sum():>+8.2f} {pf:>6.2f}")
else:
    print("no trades to bucket")
