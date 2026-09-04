#!/usr/bin/env python3
"""miss_learner.py — analyze backtest misses: which variable would have flipped
losing trades into winners. Parses the bot backtest log (signal/ENTER/EXIT
lines), pairs outcomes order-wise, simulates skipped signals, and reports the
honest expectancy by floor."""
import re
import numpy as np
import pandas as pd

sig_re = re.compile(r'signal (\S+ \S+) \[(\w+)\] (LONG|SHORT) \| proba=([\d.]+) r_hat=([\d.-]+) \| (TAKE|skip \(<([\d.]+)\))')
ent_re = re.compile(r'ENTER (\S+ \S+) (LONG|SHORT) \[(\w+)\]')
ext_re = re.compile(r'EXIT (\S+ \S+) \[(\w+)\] (LONG|SHORT) \| bracket (\w+) filled @ ([\d.]+) \| ([+-][\d.]+)R')

def parse_log(path):
    rows = []
    pending = None
    for line in open(path):
        m = sig_re.search(line)
        if m:
            t, strat, side, proba, rhat, status, floor = m.groups()
            idx = len(rows)
            rows.append({'time': t, 'side': side, 'proba': float(proba),
                         'r_hat': float(rhat), 'taken': status == 'TAKE',
                         'exit_r': np.nan, 'reason': ''})
            pending = idx if status == 'TAKE' else None
            continue
        if ent_re.search(line):
            continue
        m = ext_re.search(line)
        if m and pending is not None:
            t, strat, side, kind, px, r = m.groups()
            rows[pending]['exit_r'] = float(r)
            rows[pending]['reason'] = kind
            pending = None
    return pd.DataFrame(rows)

def simulate_skips(skips, data_path):
    data = pd.read_csv(data_path, parse_dates=['datetime']).set_index('datetime')
    h, l, c = data['high'], data['low'], data['close']
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 20, adjust=False).mean()
    STOP_ATR = 0.5
    sim_r = []
    for _, row in skips.iterrows():
        t = pd.Timestamp(row['time'], tz='UTC')
        i = data.index.get_indexer([t], method='nearest')[0]
        if data.index[i] < t:
            i = min(i + 1, len(data) - 1)
        entry = c.iloc[i]
        risk = STOP_ATR * atr.iloc[i]
        if not np.isfinite(risk) or risk <= 0:
            sim_r.append(np.nan)
            continue
        sd = 1 if row['side'] == 'LONG' else -1
        stop, tgt = entry - sd * risk, entry + sd * 2 * risk
        r = np.nan
        for j in range(i, min(i + 400, len(data))):
            if sd > 0:
                if l.iloc[j] <= stop:
                    r = -1.0
                    break
                if h.iloc[j] >= tgt:
                    r = 2.0
                    break
            else:
                if h.iloc[j] >= stop:
                    r = -1.0
                    break
                if l.iloc[j] <= tgt:
                    r = 2.0
                    break
        if np.isnan(r):
            r = sd * (c.iloc[min(i + 399, len(data) - 1)] - entry) / risk
        sim_r.append(r)
    return sim_r

def main():
    import sys
    log = sys.argv[1] if len(sys.argv) > 1 else '/tmp/backtest_misses.log'
    data_path = sys.argv[2] if len(sys.argv) > 2 else 'data/NQ_3min.csv'
    df = parse_log(log)
    taken = df[df['taken'] & df['exit_r'].notna()]
    skips = df[~df['taken']].copy()
    print(f"signals: {len(df)} | taken with outcome: {len(taken)} | skipped: {len(skips)}")
    print(f"taken WR: {(taken['exit_r'] > 0).mean() * 100:.1f}%  meanR {taken['exit_r'].mean():+.2f}  "
          f"(targets {(taken['reason'] == 'target').sum()}, stops {(taken['reason'] == 'stop').sum()})")

    print("\n--- taken trades by proba band ---")
    for lo, hi in [(0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
                   (0.50, 0.60), (0.60, 1.01)]:
        b = taken[(taken['proba'] >= lo) & (taken['proba'] < hi)]
        if len(b):
            wr = (b['exit_r'] > 0).mean() * 100
            print(f"  proba [{lo:.2f},{hi:.2f}): n={len(b):3d}  WR={wr:5.1f}%  "
                  f"meanR={b['exit_r'].mean():+.2f}")

    skips['sim_r'] = simulate_skips(skips, data_path)
    ok = skips['sim_r'].notna()
    print(f"\nskipped simulated: {ok.sum()}")
    if ok.sum():
        print(f"skipped would-WR: {(skips.loc[ok, 'sim_r'] > 0).mean() * 100:.1f}%  "
              f"meanR {skips.loc[ok, 'sim_r'].mean():+.2f}")
        for lo, hi in [(0.0, 0.20), (0.20, 0.25), (0.25, 0.30)]:
            b = skips[ok][(skips['proba'] >= lo) & (skips['proba'] < hi)]
            if len(b):
                print(f"  proba [{lo:.2f},{hi:.2f}): n={len(b):3d}  "
                      f"wouldWR={(b['sim_r'] > 0).mean() * 100:5.1f}%  "
                      f"meanR={b['sim_r'].mean():+.2f}")

    print("\n--- expectancy by floor (real outcomes + would-have) ---")
    for floor in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        rs = []
        for _, row in df.iterrows():
            if row['proba'] < floor:
                continue
            if row['taken'] and not np.isnan(row['exit_r']):
                rs.append(row['exit_r'])
            else:
                sr = skips.loc[row.name, 'sim_r']
                if not np.isnan(sr):
                    rs.append(sr)
        if rs:
            rsa = np.array(rs)
            print(f"  floor={floor:.2f}: n={len(rsa):3d}  "
                  f"WR={(rsa > 0).mean() * 100:5.1f}%  "
                  f"expectancy={rsa.mean():+.3f}R  sumR={rsa.sum():+.1f}")

if __name__ == '__main__':
    main()
