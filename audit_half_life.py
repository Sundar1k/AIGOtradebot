import os
#!/usr/bin/env python3
"""audit_half_life.py — #9 mean-reversion speed (RAmmStein half-life proxy).

Does "how fast does price snap back" separate good from bad entries? Proxied by
the rolling lag-1 autocorrelation of returns at entry:
    negative autocorr = fast mean-reversion = noise (bad for EMA-cross trend-follow)
    positive autocorr = persistence/trend (good)
Point-in-time: trailing window only, no look-ahead.
Buckets engine trades by autocorr quartile; a clean monotone split = worth a gate.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import bot
import backtest
from backtest import drive, _resolve_specs, WINDOW
from sim_broker import SimBroker


def stats(trades, label):
    r = np.array([t.r for t in trades]) if trades else np.array([])
    if len(r) == 0:
        print(f"{label}: 0 trades")
        return
    w = r[r > 0].sum()
    l = -r[r < 0].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"{label}: n={len(r)}  WR={100*(r > 0).mean():.1f}%  "
          f"meanR={r.mean():+.3f}  sumR={r.sum():+.2f}  PF={pf:.2f}")


def run(sym, start):
    config.SYMBOL = sym
    base = config.base_symbol(sym)
    tick, tick_value = _resolve_specs(sym)
    df = backtest._load(base, None)
    start_idx = max(WINDOW, int(df.index[df["time"] >= pd.Timestamp(start, tz="UTC")][0]))
    config.JUMP_ATR_MULT = 0.0
    sim = SimBroker(df, tick, tick_value=tick_value,
                    slip_ticks=config.SLIPPAGE_TICKS,
                    commission_per_side=config.COMMISSION_PER_SIDE_USD)
    ctx = bot.BotContext(sim, 0, sym, tick, tick_value, log_candles=False)
    trades = drive(ctx, sim, df, start_idx)

    closes = df["close"].to_numpy(float)
    rets = np.diff(np.log(closes))
    time_idx = {ts: i for i, ts in enumerate(df["time"])}
    W = 100                              # trailing returns for the autocorr window
    rows = []
    for t in trades:
        i = time_idx.get(t.entry_time)
        if i is None or i - 1 < W:
            continue
        w = rets[i - W - 1:i - 1]        # returns strictly before the entry bar
        sd = np.std(w)
        if sd <= 0 or not np.isfinite(sd):
            continue
        ac = float(np.corrcoef(w[:-1], w[1:])[0, 1])
        rows.append((ac, t.r))
    return trades, rows


for sym in sys.argv[1:] or ["NQ", "ES", "GC"]:
    trades, rows = run(sym, "2026-04-21")
    stats(trades, f"\n{sym} engine")
    if not rows:
        continue
    acs = np.array([x[0] for x in rows])
    rs = np.array([x[1] for x in rows])
    q = np.quantile(acs, [0.25, 0.5, 0.75])
    print(f"  lag-1 autocorr quartiles: {q[0]:+.3f} / {q[1]:+.3f} / {q[2]:+.3f}")
    edges = [-2.0, q[0], q[1], q[2], 2.0]
    for k in range(4):
        lo, hi = edges[k], edges[k + 1]
        m = (acs >= lo) & (acs <= hi)
        if not m.any():
            continue
        rr = rs[m]
        w = rr[rr > 0].sum()
        l = -rr[rr < 0].sum()
        pf = w / l if l > 0 else float("inf")
        tag = "mean-revert" if hi < 0 else ("trend" if lo > 0 else "mixed")
        print(f"  ac[{lo:+.2f},{hi:+.2f}] ({tag:>11}): n={m.sum():>3} "
              f"WR={100*(rr > 0).mean():.0f}%  meanR={rr.mean():+.3f}  PF={pf:.2f}")
