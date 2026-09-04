"""quality_score.py — veto-quality scoring pass for the dataset (spec US3/FR-006).

For every TAKE signal in the replayed dataset, build the exact live state line
(supervisor.build_state_line — the byte-identical 7B input format) from the
trailing 300 bars and ask the RUNNING veto service (:8765 /score_batch) for its
quality 1-10. Writes veto_quality back into the per-symbol jsonl files.

Scope is deliberately TAKE-only: the candidate selects WITHIN the baseline
funnel (baseline AND quality >= q), so quality of skipped signals is
irrelevant — scoring them would waste GPU for nothing (pre-registered in
spec.md US3). Rows left unscored keep veto_quality=0 and are rejected by every
candidate q (spec edge case).

Leakage note (recorded in verdict.md): quality comes from the CURRENT 7B
adapter (trained through 2026-08-16), so historical scores are
as-deployed — the selector as it would run today, not a point-in-time model.
The clean forward slice is live/paper.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import config                                # noqa: E402

OUT_DIR = os.path.join(BASE, "selection_validator", "data")
VETO_URL = os.environ.get("AUTOTRADE_VETO_URL", "http://127.0.0.1:8765")
BATCH = 32
TIMEOUT = 180


def _bars_at(symbol: str, ts: pd.Timestamp) -> pd.DataFrame:
    """Trailing 300 bars <= ts — the live broker window geometry (limit=300)."""
    path = os.path.join(BASE, "data", f"{config.base_symbol(symbol)}_{config.TIMEFRAME_MIN}min.csv")
    df = pd.read_csv(path).rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[df["time"] <= ts]
    return df.tail(300).reset_index(drop=True)


def _state_line(symbol: str, ts: pd.Timestamp) -> str:
    from supervisor import build_state_line
    bars = _bars_at(symbol, ts)
    if len(bars) < 30:
        return ""
    return build_state_line(bars.rename(columns={"time": "time"}), symbol)


def score_rows(rows: list[dict], retries: int = 3) -> list[int]:
    """Score a chunk via the running veto service; returns qualities (0 on error)."""
    texts = [r["state"] for r in rows]
    for attempt in range(retries):
        try:
            r = requests.post(f"{VETO_URL}/score_batch_v2", json={"texts": texts}, timeout=TIMEOUT)
            r.raise_for_status()
            quals = r.json().get("qualities", [])
            if len(quals) == len(rows):
                return [int(q or 0) for q in quals]
        except Exception as e:
            print(f"  ⚠ score_batch attempt {attempt+1}/{retries} failed: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return [0] * len(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None,
                    help="comma-separated symbols to score (default: all present)")
    args = ap.parse_args()
    want = {s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()}

    # health check
    try:
        h = requests.get(f"{VETO_URL}/health", timeout=5)
        print(f"veto service: HTTP {h.status_code}", flush=True)
    except Exception as e:
        print(f"veto service unreachable ({e}) — cannot score. "
              f"Start veto.service first.", flush=True)
        sys.exit(1)

    rows_all = []
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.startswith("signals_"):
            continue
        sym = fname.replace("signals_", "").replace(".jsonl", "")
        if want and sym not in want:
            continue
        path = os.path.join(OUT_DIR, fname)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows_all.append(json.loads(line))
    take_rows = [r for r in rows_all if r.get("take") and not r.get("jump")]
    print(f"total signals: {len(rows_all)} | funnel-taken: {len(take_rows)}", flush=True)
    if not take_rows:
        print("nothing to score", flush=True)
        return

    # build state lines (cache per symbol to avoid re-reading CSVs per row)
    cache: dict[str, pd.DataFrame] = {}
    for r in take_rows:
        sym = r["symbol"]
        if sym not in cache:
            path = os.path.join(BASE, "data",
                                f"{config.base_symbol(sym)}_{config.TIMEFRAME_MIN}min.csv")
            df = pd.read_csv(path).rename(columns={"datetime": "time"})
            df["time"] = pd.to_datetime(df["time"], utc=True)
            cache[sym] = df
        ts = pd.Timestamp(r["ts"], tz="UTC")
        win = cache[sym][cache[sym]["time"] <= ts].tail(300).reset_index(drop=True)
        if len(win) < 30:
            r["state"] = ""
        else:
            from supervisor import build_state_line
            r["state"] = build_state_line(win, sym)

    # score in chunks
    t0 = time.time()
    done = 0
    for i in range(0, len(take_rows), BATCH):
        chunk = take_rows[i:i + BATCH]
        quals = score_rows(chunk)
        for r, q in zip(chunk, quals):
            r["veto_quality"] = q
        done += len(chunk)
        if done % (BATCH * 10) == 0 or done == len(take_rows):
            el = time.time() - t0
            print(f"  scored {done}/{len(take_rows)} ({100*done/len(take_rows):.0f}%) "
                  f"| {el/60:.1f} min elapsed", flush=True)

    # write back (merge quality into the jsonl files by row identity)
    idx = {(r["symbol"], r["ts"], r["strategy"], r["proba"]): r["veto_quality"]
           for r in take_rows}
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.startswith("signals_"):
            continue
        sym = fname.replace("signals_", "").replace(".jsonl", "")
        if want and sym not in want:
            continue          # never touch files a replay may still be appending
        path = os.path.join(OUT_DIR, fname)
        out_lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = idx.get((d["symbol"], d["ts"], d["strategy"], d["proba"]))
                if q is not None:
                    d["veto_quality"] = q
                out_lines.append(json.dumps(d))
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + "\n")
    print(f"=== quality pass complete: {len(take_rows)} scored in "
          f"{(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
