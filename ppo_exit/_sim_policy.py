"""Drive the REAL 3-min PPO policy through real NQ bars (the same sim the
policy was trained on: TrailExitSim + NumpyMlpPolicy, tanh, live knobs from
exit_configs.json[3]) and print the stop path trade-by-trade."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
config.apply_exit_config(3)   # live knobs: ACTIVATE_R=2.0 GIVEBACK_R=0.75 STOP_ATR=0.5
print(f"live knobs: ACTIVATE_R={config.ACTIVATE_R} GIVEBACK_R={config.GIVEBACK_R} "
      f"STOP_ATR={config.STOP_ATR} MAX_HOLD={config.MAX_HOLD if hasattr(config, 'MAX_HOLD') else 80}")

from ppo_exit.trail_exit_env import (build_arrays, build_catalog, TrailExitSim,
                                     NumpyMlpPolicy, TRAIL_MULTS)

df = pd.read_csv("data/NQ_3min.csv", parse_dates=["datetime"])
print(f"bars: {len(df)}  ({df['datetime'].iloc[0]} .. {df['datetime'].iloc[-1]})")

arr = build_arrays(df)
catalog = build_catalog(arr)
print(f"SuperTrend-flip trades in catalog: {len(catalog)}")

policy = NumpyMlpPolicy.load(config.POLICY_PATH)
rng = np.random.default_rng(42)
idx = rng.choice(len(catalog), size=24, replace=False)

def simulate(entry_idx, sign):
    sim = TrailExitSim(arr)
    obs = sim.reset(int(entry_idx), int(sign))
    trail = []
    exit_kind = "?"
    while True:
        a = policy.action(obs)
        mult = TRAIL_MULTS[a]
        prior_stop = sim.stop
        trail.append((sim.i, sim.bars_held, float(sim.entry), float(sim.stop),
                      mult, float(sim._obs()[0]), float(sim._obs()[1]), sim.peak_R))
        obs, reward, done, info = sim.step(a)
        if done:
            # classify: fill at prior_stop => resting-stop fill; else market close
            fill = sim.entry + sim.sign * float(info["realized_R"]) * sim.risk
            exit_kind = ("resting-stop fill" if abs(fill - prior_stop) < 1e-6
                         else "maxhold timeout" if sim.bars_held >= 80
                         else "market close (tightened floor crossed)")
            return sim, info["realized_R"], trail, exit_kind

print("\n=== 3 trades where the trail ACTIVATED (peak >= ACTIVATE_R) ===")
# find activated trades: step the sim with a fixed 2.0x action (peak_R is
# action-independent — it tracks bar extremes) and keep peak >= ACTIVATE_R
def peak_only(j):
    entry_idx, sign = catalog[j]
    sim = TrailExitSim(arr)
    obs = sim.reset(int(entry_idx), int(sign))
    while True:
        obs, _r, done, _i = sim.step(3)      # 2.0x ATR
        if done:
            return sim.peak_R
cands = [j for j in rng.choice(len(catalog), size=400, replace=False)
         if peak_only(j) >= config.ACTIVATE_R]
rng2 = np.random.default_rng(7)
for j in rng2.choice(cands, size=min(3, len(cands)), replace=False):
    entry_idx, sign = catalog[j]
    sim, R, trail, kind = simulate(entry_idx, sign)
    ts = df["datetime"].iloc[entry_idx]
    print(f"\nTRADE: {'LONG' if sign > 0 else 'SHORT'} entry @ {ts} close={trail[0][2]:.2f} "
          f"-> realized {R:+.2f}R ({sim.bars_held} bars, {kind})")
    prev_mult = None
    n = len(trail)
    for t_i, (i, bh, entry, stop, mult, unreal, mfe, peak) in enumerate(trail):
        if mult != prev_mult or t_i >= n - 3:
            print(f"   bar {i:5d} held={bh:2d} mult={mult:4.1f}x  stop={stop:9.2f}  "
                  f"unreal={unreal:+6.2f}R  mfe={mfe:+6.2f}R  peak={peak:+5.2f}R")
            prev_mult = mult

print("\n=== 60-trade summary (random catalog sample) ===")
idx = rng.choice(len(catalog), size=60, replace=False)
Rs, bars, kinds = [], [], []
for j in idx:
    entry_idx, sign = catalog[j]
    sim, R, trail, kind = simulate(entry_idx, sign)
    Rs.append(R)
    bars.append(sim.bars_held)
    kinds.append(kind)
Rs = np.array(Rs)
print(f"  realized R: {Rs.mean():+.3f} mean | median {np.median(Rs):+.3f} | "
      f"std {Rs.std():.2f} | sum {Rs.sum():+.2f}")
print(f"  wins: {(Rs > 0).sum()}/60 ({(Rs > 0).mean():.0%})")
print(f"  avg bars held: {np.mean(bars):.1f} (max {max(bars)})")
from collections import Counter
for k, c in Counter(kinds).most_common():
    print(f"  {k}: {c}")
