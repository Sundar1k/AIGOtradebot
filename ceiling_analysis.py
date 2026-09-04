#!/usr/bin/env python3
"""ceiling_analysis.py — expectancy under a proba CEILING, computed from a
backtest log (every taken signal already has its realized R). A ceiling C
means: skip signals with proba > C. Only taken signals matter for the
counterfactual (skipped ones were never graded)."""
import re
import sys
import numpy as np
import pandas as pd

sig_re = re.compile(r'signal (\S+ \S+) \[(\w+)\] (LONG|SHORT) \| proba=([\d.]+) r_hat=([\d.-]+) \| (TAKE|skip \(<([\d.]+)\))')
ext_re = re.compile(r'EXIT (\S+ \S+) \[(\w+)\] (LONG|SHORT) \| bracket (\w+) filled @ ([\d.]+) \| ([+-][\d.]+)R')

def parse(path):
    rows, pending = [], None
    for line in open(path):
        m = sig_re.search(line)
        if m:
            rows.append({'year': m.group(1)[:4], 'proba': float(m.group(4)),
                         'exit_r': np.nan})
            pending = len(rows) - 1 if m.group(6) == 'TAKE' else None
            continue
        m = ext_re.search(line)
        if m and pending is not None:
            rows[pending]['exit_r'] = float(m.group(6))
            pending = None
    return pd.DataFrame(rows)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/backtest_5y.log'
    df = parse(path)
    taken = df[df['exit_r'].notna()]
    print(f"taken: {len(taken)}  (no ceiling baseline: "
          f"WR={(taken['exit_r']>0).mean()*100:.1f}%  expectancy={taken['exit_r'].mean():+.3f}R  "
          f"sumR={taken['exit_r'].sum():+.1f})\n")

    print("ceiling | n    | WR    | expectancy | sumR   (skip proba > ceil)")
    for ceil in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]:
        keep = taken[taken['proba'] <= ceil]
        if len(keep):
            r = keep['exit_r']
            print(f"  {ceil:.2f}  | {len(keep):4d} | {100*(r>0).mean():5.1f}% | "
                  f"{r.mean():+.3f}R   | {r.sum():+.1f}")

    print("\nby year, ceiling 0.50 vs none:")
    for y in sorted(taken['year'].unique()):
        b = taken[taken['year'] == y]
        k = b[b['proba'] <= 0.50]
        print(f"  {y}: n={len(b):4d} WR={100*(b['exit_r']>0).mean():4.1f}% exp={b['exit_r'].mean():+.3f}R  "
              f"-> ceil0.50 n={len(k):4d} WR={100*(k['exit_r']>0).mean():4.1f}% "
              f"exp={k['exit_r'].mean():+.3f}R")

if __name__ == '__main__':
    main()
