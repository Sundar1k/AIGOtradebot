"""dedup_signals.py — remove duplicate signal rows before analysis.

Duplicate (symbol, ts, strategy, direction, entry, proba) rows can appear when
two replay processes appended to the same file (the sequential NQ process
re-duplicated ES after the parallel split, 2026-08-30). Idempotent and safe on
clean files: keeps the FIRST occurrence of each key, preserves order.
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "selection_validator", "data")


def dedup_file(path: str) -> tuple[int, int]:
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("symbol"), r.get("ts"), r.get("strategy"),
                   r.get("direction"), round(float(r.get("entry", 0)), 2),
                   r.get("proba"))
            if key in seen:
                continue
            seen.add(key)
            out.append(json.dumps(r))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return len(seen), len(out)


def main():
    total_before = total_after = 0
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.startswith("signals_"):
            continue
        path = os.path.join(OUT_DIR, fname)
        before = sum(1 for _ in open(path) if _.strip())
        n_unique, _ = dedup_file(path)
        total_before += before
        total_after += n_unique
        if before != n_unique:
            print(f"{fname}: {before} -> {n_unique} rows ({before - n_unique} dupes removed)")
    print(f"total: {total_before} -> {total_after}")


if __name__ == "__main__":
    sys.exit(main() or 0)
