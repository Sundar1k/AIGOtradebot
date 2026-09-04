#!/usr/bin/env python3
"""Quick proba-band analysis on a backtest log (partial or full)."""
import re
import sys
import numpy as np
import pandas as pd

sig_re = re.compile(r'signal (\S+ \S+) \[(\w+)\] (LONG|SHORT) \| proba=([\d.]+) r_hat=([\d.-]+) \| (TAKE|skip \(<([\d.]+)\))')
ext_re = re.compile(r'EXIT (\S+ \S+) \[(\w+)\] (LONG|SHORT) \| bracket (\w+) filled @ ([\d.]+) \| ([+-][\d.]+)R')

def analyze(path):
    rows, pending = [], None
    for line in open(path):
        m = sig_re.search(line)
        if m:
            rows.append({'proba': float(m.group(4)), 'taken': m.group(6) == 'TAKE',
                         'exit_r': np.nan})
            pending = len(rows) - 1 if m.group(6) == 'TAKE' else None
            continue
        m = ext_re.search(line)
        if m and pending is not None:
            rows[pending]['exit_r'] = float(m.group(6))
            pending = None
    df = pd.DataFrame(rows)
    taken = df[df['taken'] & df['exit_r'].notna()]
    print(f"signals={len(df)} taken={len(taken)}")
    for lo, hi in [(0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
                   (0.50, 0.60), (0.60, 1.01)]:
        b = taken[(taken['proba'] >= lo) & (taken['proba'] < hi)]
        if len(b):
            wr = (b['exit_r'] > 0).mean() * 100
            print(f"  proba [{lo:.2f},{hi:.2f}): n={len(b):3d}  "
                  f"WR={wr:5.1f}%  meanR={b['exit_r'].mean():+.2f}")
    return df, taken

if __name__ == '__main__':
    df, taken = analyze(sys.argv[1] if len(sys.argv) > 1 else '/tmp/backtest_5y.log')
