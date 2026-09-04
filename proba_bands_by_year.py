#!/usr/bin/env python3
"""proba_bands_by_year.py — WR by proba band and year, from a backtest log."""
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
            t = m.group(1)[:4]
            rows.append({'year': t, 'proba': float(m.group(4)),
                         'taken': m.group(6) == 'TAKE', 'exit_r': np.nan})
            pending = len(rows) - 1 if m.group(6) == 'TAKE' else None
            continue
        m = ext_re.search(line)
        if m and pending is not None:
            rows[pending]['exit_r'] = float(m.group(6))
            pending = None
    df = pd.DataFrame(rows)
    taken = df[df['taken'] & df['exit_r'].notna()]
    print(f"signals={len(df)} taken={len(taken)}\n")
    years = sorted(taken['year'].unique())
    bands = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.01)]
    hdr = "year   n    | " + " | ".join(f"[{lo:.2f},{hi:.2f})" for lo, hi in bands)
    print(hdr)
    print("-" * len(hdr))
    for y in years:
        b = taken[taken['year'] == y]
        cells = []
        for lo, hi in bands:
            bb = b[(b['proba'] >= lo) & (b['proba'] < hi)]
            if len(bb):
                cells.append(f"n={len(bb):3d} wr={100*(bb['exit_r']>0).mean():4.0f}%")
            else:
                cells.append("   --    ")
        print(f"{y}  {len(b):4d} | " + " | ".join(cells))

if __name__ == '__main__':
    analyze(sys.argv[1] if len(sys.argv) > 1 else '/tmp/backtest_5y.log')
