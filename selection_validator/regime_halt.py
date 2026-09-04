"""regime_halt.py — point-in-time replay of the LIVE halt rules (spec 02).

Replays the edge-monitor realized-edge halt + the chop gate over the cycle-1
scored-signal dataset, book-level and chronological, using the EXACT live
parameters (FR-004, zero tuning):

  edge rule:  trailing WINDOW=15 closed candidate trades; halt iff
              P(meanR<0) > HALT_P=0.90 (bootstrap 10k, seed 42 — same
              bootstrap_p_lt as the live edge_monitor) OR (WR<0.30 AND
              meanR<0); COOLDOWN_H=24 then resume-to-test with a fresh
              window; re-halt if still losing.
  chop gate:  chop_gate.should_block(bars_at(ts)) — ATR14/ATR100 >= 1.0,
              MIN_BARS=120, fail-open.

Point-in-time only: the window holds trades closed BEFORE the signal ts;
no look-ahead. Cold start (<15 closed trades) takes everything — mirrors
live warmup. The regime HMM is excluded (no historical states — spec).

Live loop/config are never touched (FR-006).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)                      # repo root — chop_gate, edge_monitor

import chop_gate                              # noqa: E402
from edge_monitor import bootstrap_p_lt        # noqa: E402  P(mean<0), 10k seed 42
from selection_validator import harness        # noqa: E402
DATA_GLOB = os.path.join(BASE, "selection_validator", "data", "signals_*.jsonl")

# ── LIVE parameters (FR-004) — do not tune ───────────────────────────────
WINDOW = 15
HALT_P = 0.90
HALT_WR = 0.30
COOLDOWN_H = 24
CHOP_MAX = 1.0
TRAIL_BARS = 300          # broker get_bars limit — live window geometry


@dataclass
class HaltDecision:
    ts: str
    symbol: str
    go: bool
    edge_halt: bool
    chop_halt: bool
    window_n: int
    window_wr: float
    window_mean_r: float
    p_lt0: float
    reason: str = ""


class HaltSimulator:
    """Book-level edge-monitor + chop simulation over chronological signals."""

    def __init__(self, take_df: pd.DataFrame, cooldown_h: float = COOLDOWN_H,
                 halt_p: float = HALT_P, halt_wr: float = HALT_WR):
        self.rows = take_df.sort_values("ts").reset_index(drop=True)
        self.cooldown_h = cooldown_h
        self.halt_p = halt_p
        self.halt_wr = halt_wr
        self.trades: deque = deque(maxlen=200)     # r of candidate-closed trades
        self.state = "normal"                      # normal | halt
        self.halted_at: pd.Timestamp | None = None
        self.decisions: list[HaltDecision] = []
        self.blocks_edge = 0
        self.blocks_chop = 0
        self._bars_cache: dict[str, pd.DataFrame] = {}
        self._chop_cache: dict[tuple, bool] = {}     # (symbol, ts) -> chop blocked

    # ── chop gate (cached per symbol CSV) ────────────────────────────────
    def _chop_blocked(self, row) -> bool:
        sym = row["symbol"]
        ts = row["ts"]
        key = (sym, str(ts))
        if key in self._chop_cache:
            return self._chop_cache[key]
        try:
            if sym not in self._bars_cache:
                path = os.path.join(BASE, "data", f"{sym}_3min.csv")
                df = pd.read_csv(path).rename(columns={"datetime": "time"})
                df["time"] = pd.to_datetime(df["time"], utc=True)
                self._bars_cache[sym] = df
            df = self._bars_cache[sym]
            win = df[df["time"] <= ts].tail(TRAIL_BARS).reset_index(drop=True)
            blocked, _why = chop_gate.should_block(win, max_ratio=CHOP_MAX)
        except Exception:
            blocked = False                        # fail-open (live semantics)
        self._chop_cache[key] = blocked
        return blocked

    # ── edge rule state machine ──────────────────────────────────────────
    def _edge_halted(self, ts: pd.Timestamp) -> tuple[bool, int, float, float, float]:
        if self.state == "halt":
            if self.halted_at is not None and \
               (ts - self.halted_at).total_seconds() >= self.cooldown_h * 3600:
                # cooldown elapsed — resume to test with the fresh window
                self.state = "normal"
                self.halted_at = None
            else:
                return True, len(self.trades), 0.0, 0.0, 1.0
        n = len(self.trades)
        if n < WINDOW:
            return False, n, 0.0, 0.0, 0.5            # cold start — take
        rs = np.array(list(self.trades)[-WINDOW:], dtype=float)
        wr = float((rs > 0).mean())
        mean_r = float(rs.mean())
        p_lt0 = bootstrap_p_lt(rs, 0.0)               # P(meanR<0), 10k seed 42
        losing = p_lt0 > self.halt_p or (wr < self.halt_wr and mean_r < 0)
        if losing:
            self.state = "halt"
            self.halted_at = ts
            self.trades.clear()                       # fresh window on resume
        return losing, n, wr, mean_r, p_lt0

    # ── run ─────────────────────────────────────────────────────────────
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (decisions_df, candidate_trades_df)."""
        cand_rows = []
        for row in self.rows.to_dict("records"):
            ts = pd.Timestamp(row["ts"])          # already tz-aware (UTC)
            chop = self._chop_blocked(row)
            edge, n, wr, mean_r, p_lt0 = self._edge_halted(ts)
            go = not (chop or edge)
            if edge:
                self.blocks_edge += 1
            if chop:
                self.blocks_chop += 1
            self.decisions.append(HaltDecision(
                ts=str(ts), symbol=row["symbol"], go=go,
                edge_halt=edge, chop_halt=chop,
                window_n=n, window_wr=round(wr, 4),
                window_mean_r=round(mean_r, 4), p_lt0=round(p_lt0, 4),
                reason=("chop" if chop else "") + ("|edge" if edge else ""),
            ))
            if go:
                r = row.get("outcome_r")
                cand_rows.append({**row, "halted": False})
                if r is not None:
                    self.trades.append(float(r))      # window = candidate trades
        return (pd.DataFrame(self.decisions),
                pd.DataFrame(cand_rows))


def build_take_rows() -> pd.DataFrame:
    from selection_validator.dataset import load_rows
    df = load_rows(*glob.glob(DATA_GLOB))
    m = df["take"].astype(bool) & ~df["jump"].astype(bool)
    return df[m].reset_index(drop=True)


def compare(take_df: pd.DataFrame, cand_df: pd.DataFrame, boundary: str = "2026-08-01"):
    """Full-period + August-slice stats and bootstrap P(ΔavgR>0)."""
    from selection_validator.selectors import evaluate, BaselineSelector
    b_all = evaluate(take_df, BaselineSelector())
    c_all = evaluate(cand_df, BaselineSelector())
    b = pd.Timestamp(boundary, tz="UTC")
    t_aug = take_df[take_df["ts"] >= b]
    c_aug = cand_df[cand_df["ts"] >= b]
    b_aug = evaluate(t_aug, BaselineSelector())
    c_aug_ = evaluate(c_aug, BaselineSelector())
    base_rs = take_df["outcome_r"].dropna()
    cand_rs = cand_df["outcome_r"].dropna()
    p_lt0 = harness.bootstrap_diff(base_rs, cand_rs)  # P(Δ<0)
    return {
        "full": {"baseline": b_all, "candidate": c_all,
                 "p_delta_gt0": round(1 - p_lt0, 4)},
        "august": {"baseline": b_aug, "candidate": c_aug_},
    }


def decide_verdict(comp: dict) -> tuple[str, dict]:
    f, a = comp["full"], comp["august"]
    base, cand = f["baseline"], f["candidate"]
    p = f["p_delta_gt0"]
    n_base, n_cand = base["closed"], cand["closed"]
    aug_ok = a["candidate"]["avg_r"] is not None and a["baseline"]["avg_r"] is not None \
        and a["candidate"]["avg_r"] > a["baseline"]["avg_r"]
    checks = {
        "a_p_gt_095": p > 0.95,
        "b_pf_cand_ge_base": (cand["pf"] or 0) >= (base["pf"] or 0),
        "c_aug_improved": bool(aug_ok),
        "d_n_cand_ge_half": n_cand >= 0.5 * n_base,
    }
    if all(checks.values()):
        return "GO", checks
    if not checks["d_n_cand_ge_half"]:
        return "KILL", {**checks, "why": "rule halts too much (n_cand < half of baseline)"}
    if p <= 0.5 or (cand["pf"] or 0) < (base["pf"] or 0) or not aug_ok:
        return "KILL", checks
    return "INCONCLUSIVE", checks      # directionally better but insignificant


if __name__ == "__main__":
    import sys
    from selection_validator.dedup_signals import main as dedup
    dedup()
    take = build_take_rows()
    print(f"funnel-taken: {len(take)}", flush=True)
    sim = HaltSimulator(take)
    decisions, cand = sim.run()
    print(f"sim done: {len(cand)} candidate trades | edge blocks "
          f"{sim.blocks_edge} | chop blocks {sim.blocks_chop}", flush=True)
    comp = compare(take, cand)
    verdict, checks = decide_verdict(comp)
    print(json.dumps({"verdict": verdict, "checks": checks}, indent=1))
    print("FULL  baseline:", json.dumps(comp["full"]["baseline"]))
    print("FULL  candidate:", json.dumps(comp["full"]["candidate"]))
    print("AUG   baseline:", json.dumps(comp["august"]["baseline"]))
    print("AUG   candidate:", json.dumps(comp["august"]["candidate"]))
