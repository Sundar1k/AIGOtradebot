#!/usr/bin/env python3
"""eval_ppo_policy.py — benchmark one exit policy (.npz) on the NQ holdout,
reusing train_ppo_exit's EXACT setup (proba-floor filter + time split), so the
A/B between the symmetric and asymmetric policies is apples-to-apples."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from ppo_exit.train_ppo_exit import (
    build_arrays, build_catalog, eval_policy, eval_fixed_2r,
    NumpyMlpPolicy, HOLDOUT_FRAC,
)
from ppo_exit import precompute_proba as pp

NPZ = sys.argv[1]
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "NQ_3min.csv")

df = pd.read_csv(CSV)
arr = build_arrays(df)
catalog = build_catalog(arr)
if FLOOR > 0:
    proba = pp.read_cache(df, catalog, CSV)
    if proba is None:
        pp.grade_in_subprocess(CSV)
        proba = pp.read_cache(df, catalog, CSV)
    if proba is None:
        raise SystemExit("proba grading failed")
    catalog = catalog[proba >= FLOOR]

cut = int(len(df) * (1 - HOLDOUT_FRAC))
hold_cat = catalog[catalog[:, 0] >= cut]
print(f"holdout entries (proba>={FLOOR}): {len(hold_cat)}")

pol = NumpyMlpPolicy.load(NPZ)
m, w, p, n = eval_policy(arr, hold_cat, pol.action)
print(f"policy {os.path.basename(NPZ)}:  meanR={m:+.3f}  WR={w:.1%}  PF={p:.2f}  n={n}")
m, w, p, n = eval_fixed_2r(arr, hold_cat)
print(f"fixed 2R baseline:         meanR={m:+.3f}  WR={w:.1%}  PF={p:.2f}  n={n}")
