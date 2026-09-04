import os
#!/usr/bin/env python3
"""audit_veto.py — does the LLM veto actually improve selection?

A/B on the SAME blind OOS window, SAME live pipeline (handle_bar + SimBroker +
PPO exit + slippage):
  - engine-only (veto_fn=None)
  - engine + LLM veto (POST state line to the GPU sidecar :8765/decide,
    PASS only when the model's action matches the engine side)

The veto here is the LLM gate ONLY (point-in-time: it sees only the state line
built from the historical bars). The regime gate + news blackout are EXCLUDED
because they read CURRENT state (regime file / calendar) — applying them to
historical bars would be look-ahead.

Run AFTER audit_vol_jump.py so the veto sidecar is warm. Slow: each signal is a
GPU call (17-32s cold).
"""
import sys
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import bot
import supervisor
import backtest
from backtest import drive, _resolve_specs, WINDOW
from sim_broker import SimBroker

VETO_URL = "http://127.0.0.1:8765/decide"
SYM = sys.argv[1] if len(sys.argv) > 1 else "NQ"
START = sys.argv[2] if len(sys.argv) > 2 else "2026-04-21"


def make_llm_veto():
    """LLM-only veto: build the v1 state line, ask the sidecar, agree iff the
    model's action matches the engine side. Fail-closed (sidecar down = block)."""
    def veto(s, sig, bars):
        try:
            state = supervisor.build_state_line(bars, config.SYMBOL)
            r = requests.post(VETO_URL, json={"text": state}, timeout=120)
            r.raise_for_status()
            action = r.json().get("action", "NO TRADE")
        except Exception as e:
            return False, f"veto error {e}"
        want = "BUY" if sig.direction > 0 else "SELL"
        return action == want, action
    return veto


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


def run(veto_fn):
    config.JUMP_ATR_MULT = 0.0
    sim = SimBroker(df, tick, tick_value=tick_value,
                    slip_ticks=config.SLIPPAGE_TICKS,
                    commission_per_side=config.COMMISSION_PER_SIDE_USD)
    ctx = bot.BotContext(sim, account_id=0, contract_id=SYM, tick_size=tick,
                         tick_value=tick_value, log_candles=False)
    ctx.veto_fn = veto_fn
    return drive(ctx, sim, df, start_idx)


config.SYMBOL = SYM
base = config.base_symbol(SYM)
tick, tick_value = _resolve_specs(SYM)
df = backtest._load(base, None)
start_idx = max(WINDOW, int(df.index[df["time"] >= pd.Timestamp(START, tz="UTC")][0]))

# health check
try:
    h = requests.get("http://127.0.0.1:8765/health", timeout=5).json()
    print(f"veto sidecar: {h}", flush=True)
except Exception as e:
    print(f"⚠ veto sidecar unreachable: {e}", flush=True)

print(f"=== VETO A/B | {SYM} | {START} -> {df['time'].iloc[-1].date()} | "
      f"{len(df) - start_idx} bars ===", flush=True)

print("\n--- engine-only ---", flush=True)
engine = run(None)
stats(engine, "engine-only")

print("\n--- engine + LLM veto (this may take 10-30 min) ---", flush=True)
vetoed = run(make_llm_veto())
stats(vetoed, "engine+veto")
