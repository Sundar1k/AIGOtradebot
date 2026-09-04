import json, glob
from datetime import datetime, timezone
def parse(ts):
    s = ts.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
for f in sorted(glob.glob('selection_validator/data/signals_*.jsonl')):
    tot = poll = submin = 0
    keys_ok = set()
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            tot += 1
            ts = None
            for k in ('time', 'bar_time', 'ts', 'entry_ts', 'entry_time', 'entry'):
                v = r.get(k)
                if isinstance(v, str) and v:
                    ts = v
                    keys_ok.add(k)
                    break
            if ts:
                d = parse(ts)
                if d:
                    if datetime(2026, 8, 21, 16, 39, tzinfo=timezone.utc) <= d <= datetime(2026, 8, 25, 11, 30, tzinfo=timezone.utc):
                        poll += 1
                    if d.second != 0 or d.minute % 3 != 0:
                        submin += 1
    print(f, 'total=%d poll-window=%d submin/non3m=%d timekey=%s' % (tot, poll, submin, sorted(keys_ok)[:1]))
