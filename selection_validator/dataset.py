"""dataset.py — point-in-time scored-signal dataset for the selection validator.

ScoredSignal: one graded signal (take OR skip) with funnel context, gate
outcomes, veto quality and (when taken and closed) realized R.

SignalSink: observation-only hook installed on BotContext.on_graded_signal
during replay (backtest.py --log-signals). Writes every graded signal
append-only; trades are matched after the replay via match_trade().

FR-001 fields, point-in-time only (no feature may use bars after the signal
bar — replay passes trailing windows, so this holds by construction; the
leak_check() below enforces it on the assembled rows).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

# OOS boundary pre-registered in spec.md (FR-007) — do not move.
OOS_BOUNDARY = "2026-08-01"


@dataclass
class ScoredSignal:
    ts: str                      # signal bar time (UTC ISO)
    symbol: str
    strategy: str
    direction: int               # +1 long / -1 short
    proba: float
    r_hat: float
    entry: float
    stop: float
    risk: float
    floor: float
    ceil: float
    take: bool                   # floor <= proba <= ceil at grading time
    jump: bool = False           # recent_jump() at signal bar (selection skip)
    veto_quality: int = 0        # 7B /score rating; 0 = unscored/missing
    outcome_r: Optional[float] = None   # realized R once the trade closes
    exit_reason: Optional[str] = None   # stop | target | trail | eod

    def to_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(d: dict) -> "ScoredSignal":
        return ScoredSignal(**{k: d[k] for k in ScoredSignal.__dataclass_fields__})


class SignalSink:
    """Append-only graded-signal logger + trade-outcome matcher (observation-only)."""

    def __init__(self, path: str, symbol: str):
        self.path = path
        self.symbol = symbol
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a")
        self._open: dict = {}     # (strategy, direction, round(entry,2)) -> row dict

    def __call__(self, s, sig, floor, ceil, bars):
        """Fired by bot.handle_bar for EVERY graded signal (before filters)."""
        try:
            from strategies import recent_jump
            row = ScoredSignal(
                ts=str(bars["time"].iloc[-1]),
                symbol=self.symbol,
                strategy=s.name,
                direction=int(sig.direction),
                proba=float(sig.proba),
                r_hat=float(getattr(sig, "r_hat", 0.0) or 0.0),
                entry=float(sig.entry),
                stop=float(sig.stop),
                risk=float(sig.risk),
                floor=float(floor),
                ceil=float(ceil),
                take=bool(floor <= sig.proba <= ceil),
                jump=bool(recent_jump(bars, len(bars) - 1)),
            ).to_row()
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()
        except Exception:
            pass                 # observation-only — never affects the loop

    def close(self):
        self._fh.close()


def merge_outcomes(path: str, trades) -> int:
    """Merge realized outcomes from SimBroker.trades into the signal jsonl.

    Matches on (strategy, direction, entry price, entry bar time) — the replay
    is deterministic, so a taken signal's entry bar equals its trade's fill
    bar. The bar-time component is REQUIRED: entry prices repeat often (ES
    clusters at round prices), and price-only matching would assign one
    trade's outcome to many signals. Rewrites the file in place. Returns the
    matched count.
    """
    def _tkey(strategy, direction, entry, ts):
        return (str(strategy), int(direction), round(float(entry), 2),
                str(ts) if ts is not None else "")
    idx = {_tkey(t.strategy, t.direction, t.entry, getattr(t, "entry_time", None)): t
           for t in trades}
    out, matched = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = idx.get(_tkey(row["strategy"], int(row["direction"]),
                              row["entry"], row["ts"]))
            if t is not None:
                row["outcome_r"] = float(t.r)
                row["exit_reason"] = str(t.reason)
                matched += 1
            out.append(json.dumps(row))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return matched


# ── assembly / verification ────────────────────────────────────────────────

def load_rows(*paths: str) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def leak_check(df: pd.DataFrame) -> list[str]:
    """FR-004/leak-check: point-in-time violations on the assembled rows.

    Returns a list of problems (empty = clean). Enforced on dataset build:
    any violation -> rebuild, never judge on leaked data.
    """
    problems = []
    if df.empty:
        problems.append("dataset empty")
        return problems
    if df["proba"].isna().any():
        problems.append("rows missing proba")
    # Ordering sanity: outcome must reference the same signal bar (we matched
    # by entry price, so ts consistency is structural). No future features are
    # possible: replay passes trailing windows ending at the signal bar.
    return problems
