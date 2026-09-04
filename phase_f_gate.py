#!/usr/bin/env python3
"""phase_f_gate.py — Phase F: full-pipeline 15-min backtest vs 3-min baseline.

Runs the REAL bot logic (backtest.py drive → bot.handle_bar with ML-graded
lanes + PPO exit through SimBroker) on BOTH timeframes, OOS window 2025-26
(the pre-registered split), and applies the DEPLOY RULE from the master
prompt:

  switch TIMEFRAME_MIN to 15 ONLY if the 15-min full-pipeline backtest
  beats the 3-min baseline in BOTH avgR and PF out-of-sample with n>=100
  and the bootstrap CI lower bound of avgR above the baseline avgR.
  Otherwise REJECT — keep 3-min, archive the 15-min models.

Run from ~/projects/algoTraderBot:
  ./.venv/bin/python phase_f_gate.py            # 3-min baseline + 15-min arm
  ./.venv/bin/python phase_f_gate.py --tf 15    # just the 15-min arm
  ./.venv/bin/python phase_f_gate.py --tf 3     # just the 3-min baseline
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np

import config
import backtest

OOS_START = "2025-01-01"     # pre-registered OOS split (2021-24 train / 25-26 test)
SYMBOLS = ["NQ", "ES", "RTY", "YM", "GC"]
N_BOOT = 2000
SEED = 7


def _run_tf(tf: int, symbols):
    """Full-pipeline backtest per symbol on the given timeframe; returns
    the concatenated R array (OOS window only)."""
    config.TIMEFRAME_MIN = tf
    config.SYMBOL = "NQ"                       # default; per-symbol set in run
    rs = []
    for sym in symbols:
        try:
            trades = backtest.run_backtest(sym, start=OOS_START)
        except SystemExit as e:
            print(f"  {sym}: SKIP ({e})", flush=True)
            continue
        r = np.array([t.r for t in trades])
        rs.append(r)
        n = len(r)
        if n:
            wr = (r > 0).mean()
            wins, losses = r[r > 0].sum(), -r[r < 0].sum()
            pf = wins / losses if losses > 0 else float("inf")
            print(f"  {sym}: n={n}  WR={100*wr:.1f}%  avgR={r.mean():+.3f}  "
                  f"sumR={r.sum():+.1f}  PF={pf:.2f}", flush=True)
    if not rs:
        return np.array([])
    return np.concatenate(rs)


def _bootstrap_ci(R, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(R)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = R[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi, means.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=None, help="run one timeframe only")
    args = ap.parse_args()

    print("=" * 78)
    print("PHASE F — FULL-PIPELINE GATE: 15-min vs 3-min baseline (OOS 2025-26)")
    print("=" * 78)

    results = {}
    tfs = [args.tf] if args.tf else [3, 15]
    for tf in tfs:
        print(f"\n── TIMEFRAME {tf}-min ──", flush=True)
        t0 = time.time()
        R = _run_tf(tf, SYMBOLS)
        n = len(R)
        if n == 0:
            print(f"  {tf}-min: NO TRADES — cannot evaluate", flush=True)
            results[tf] = dict(n=0)
            continue
        wr = (R > 0).mean()
        wins, losses = R[R > 0].sum(), -R[R < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        lo, hi, m = _bootstrap_ci(R)
        results[tf] = dict(n=n, wr=wr, avg=R.mean(), sm=R.sum(), pf=pf,
                           ci_lo=lo, ci_hi=hi)
        print(f"  {tf}-min TOTAL: n={n}  WR={100*wr:.1f}%  avgR={R.mean():+.3f}  "
              f"sumR={R.sum():+.1f}  PF={pf:.2f}  CI95=[{lo:+.3f}, {hi:+.3f}]  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print("VERDICT")
    if 3 not in results or 15 not in results:
        print("  incomplete — need both arms for the gate.")
    else:
        b, f = results[3], results[15]
        base_avg, base_pf = b["avg"], b["pf"]
        ok_n = f["n"] >= 100
        ok_avg = f["avg"] > base_avg
        ok_pf = f["pf"] > base_pf
        ok_ci = f["ci_lo"] > base_avg
        print(f"  3-min baseline : n={b['n']}  avgR={base_avg:+.3f}  PF={base_pf:.2f}")
        print(f"  15-min arm     : n={f['n']}  avgR={f['avg']:+.3f}  PF={f['pf']:.2f}  "
              f"CI95=[{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}]")
        print(f"  gates → n>=100: {ok_n} | avgR>base: {ok_avg} | PF>base: {ok_pf} "
              f"| CI_lo>base_avg: {ok_ci}")
        if ok_n and ok_avg and ok_pf and ok_ci:
            print("  ✅ DEPLOY — switch TIMEFRAME_MIN to 15.")
        else:
            print("  ❌ REJECT — keep 3-min; archive the 15-min models for "
                  "future experiments (per pre-registered rule).")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
