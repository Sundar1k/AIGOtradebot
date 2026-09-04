import re
import pandas as pd

rows = {}
for line in open('log/bot.log'):
    if 'candle' not in line:
        continue
    m = re.search(r'candle (\S+ \S+)\s+O=([\d.]+) H=([\d.]+) L=([\d.]+) C=([\d.]+)', line)
    o = float(m.group(2))
    if 4600 < o < 4750:  # GC price band
        rows[m.group(1)] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))

ts = sorted(rows)
df = pd.DataFrame([rows[t] for t in ts], index=pd.to_datetime(ts), columns=['H', 'L', 'C'])
pre = df.loc[:'2026-08-25 07:27'].tail(20)
a = (pre['H'] - pre['L']).mean()
e = df.loc['2026-08-25 07:27', 'C']
risk = 0.5 * a
stop = e + risk
print('entry', e, '| ATR20', round(a, 2), '| stop', round(stop, 2), '| 1R =', round(risk, 2), 'pts')
post = df.loc['2026-08-25 07:30':]
stopped = False
for t, r in post.iterrows():
    if r.H >= stop:
        print('STOPPED OUT at', t.time(), 'high', r.H)
        stopped = True
        break
if not stopped:
    print('stop never hit through', post.index[-1].time())
mn = post.L.min()
print('lowest low:', mn, 'at', post.L.idxmin().time(), '=> best move', round(e - mn, 1), 'pts =', round((e - mn) / risk, 1), 'R')
print('last price:', post.iloc[-1].C)
print(post.head(12)[['C']])
