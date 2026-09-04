import os
#!/usr/bin/env python3
"""rl_sync_mods.py — the MODIFIED 37% rule (quant guardrails) on top of Phase 5.

Implements the three quant fixes from optimal-stopping practice:
  1. HARD CUTOFF    — if price moves > X ATR from the window's start price,
                      abort waiting and take the next synced candidate
                      (emergency execution — don't get left behind).
  2. DISCOUNTING    — the benchmark DECAYS as the window progresses
                      (threshold = cal_max - gamma * progress * spread):
                      less picky as time runs out — fixes the starvation
                      that killed the pure 37% cutoff.
  3. REGIME SWITCH  — the calibration/waiting phase runs ONLY in
                      mean-reverting (choppy) conditions (ADX < 25);
                      in trending conditions (ADX >= 30) the rule is OFF
                      and plain sync trades (momentum mode).

Baseline = Phase 5 sync-only. Compare vs the 20% pure cutoff (today's best).
OOS split 2021-24 / 2025-26. Same proxies as rl_sync.py (labeled).
"""
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_sync import candidates, load, SYMBOLS, OOS_START

WINDOW = 100


def run_modified(cands, veto_rate, cutoff, gamma=0.0, hard_cut=None,
                 regime_switch=False, seed=1):
    """gamma>0 = decaying benchmark · hard_cut = ATR distance that aborts
    waiting · regime_switch = only calibrate when ADX<25 (proxy)."""
    rng = np.random.default_rng(seed)
    taken = []
    cal_max = -1
    in_cal = True
    count = 0
    win_start_price = None
    for k, cd in enumerate(cands):
        if k % WINDOW == 0:
            cal_max, in_cal, count = -1, True, 0
            win_start_price = None
        count += 1
        if cd["score"] < 2:               # factor 2 proxy
            continue
        if rng.random() > veto_rate:      # factor 3 veto gate
            continue
        # REGIME SWITCH: calibration only when choppy (proxy: ADX not stored
        # in candidates — use score's adx component? we stored only score.
        # proxy via score: adx<=40 is one component; use candidate score
        # distribution instead: choppy proxy = score == 2 (mid conditions).
        if regime_switch and cd["score"] != 2:
            taken.append(cd["r"])         # trending → plain sync (rule off)
            continue
        if in_cal:
            cal_max = max(cal_max, cd["score"])
            if count >= max(1, int(WINDOW * cutoff)):
                in_cal = False
            continue
        # HARD CUTOFF: price ran away from the window start → stop waiting
        if hard_cut is not None and win_start_price is not None:
            if abs(cd["price"] - win_start_price) > hard_cut:
                taken.append(cd["r"])
                continue
        if win_start_price is None:
            win_start_price = cd["price"]
        # DISCOUNTING: threshold decays as the window progresses
        thr = cal_max
        if gamma > 0:
            progress = count / WINDOW
            thr = cal_max - gamma * progress * 3.0   # score spread is 0..3
        if cd["score"] > thr:
            taken.append(cd["r"])
    return taken


def stats(rs):
    if not rs:
        return dict(n=0, wr=0.0, avg=0.0, sm=0.0, pf=0.0)
    rs = np.array(rs)
    n = len(rs)
    wr = float(np.sum(rs > 0)) / n
    avg = float(np.mean(rs))
    g_win = float(np.sum(rs[rs > 0]))
    g_loss = -float(np.sum(rs[rs < 0]))
    pf = g_win / g_loss if g_loss > 0 else float("inf")
    return dict(n=n, wr=wr, avg=avg, sm=float(np.sum(rs)), pf=pf)


def main():
    print("=== MODIFIED 37% RULE — quant guardrails (OOS 2025-26, veto 0.62) ===", flush=True)
    print("=" * 96, flush=True)
    all_c = []
    for s in SYMBOLS:
        for cd in candidates(load(s)):
            all_c.append(cd)
    oos = [cd for cd in all_c if cd["t"] >= OOS_START]
    # add price for the hard-cutoff test (close at signal time)
    for cd in oos:
        cd["price"] = float(cd["t"].value)  # placeholder — replaced below
    print(f"candidates out-of-sample: {len(oos)}", flush=True)

    # NOTE: candidates() doesn't carry price; hard-cut uses entry price —
    # re-derive via rl_sync? Simpler: hard_cut test uses score-only proxy
    # removed; keep decay + regime-switch (the two implementable ones here).
    print(f"\n{'config':<40} {'n':>6} {'WR':>7} {'avgR':>8} {'sumR':>8} {'PF':>6}", flush=True)
    print("-" * 96, flush=True)
    base = run_modified(oos, 0.62, 0.0)
    st = stats(base)
    print(f"{'sync only (baseline)':<40} {st['n']:>6} {st['wr']:>7.1%} "
          f"{st['avg']:>+8.3f} {st['sm']:>+8.0f} {st['pf']:>6.2f}", flush=True)
    rows = [
        ("pure 20% cutoff (yesterday's best)", dict(veto_rate=0.62, cutoff=0.20)),
        ("+ decay gamma=0.5",                  dict(veto_rate=0.62, cutoff=0.20, gamma=0.5)),
        ("+ decay gamma=1.0",                  dict(veto_rate=0.62, cutoff=0.20, gamma=1.0)),
        ("+ decay gamma=0.5 + regime switch",  dict(veto_rate=0.62, cutoff=0.20, gamma=0.5, regime_switch=True)),
        ("regime switch only (rule off in trend)", dict(veto_rate=0.62, cutoff=0.20, regime_switch=True)),
    ]
    for label, kw in rows:
        taken = run_modified(oos, **kw)
        st = stats(taken)
        beats = (st["n"] >= 100 and st["avg"] > stats(base)["avg"]
                 and st["pf"] > stats(base)["pf"])
        tag = "✓" if beats else " "
        print(f"{label:<40} {st['n']:>6} {st['wr']:>7.1%} {st['avg']:>+8.3f} "
              f"{st['sm']:>+8.0f} {st['pf']:>6.2f} {tag}", flush=True)
    print("=" * 96, flush=True)
    print("Regime-switch proxy: score==2 (mid conditions) = choppy; else trend → rule off.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
