"""harness.py — pre-registered verdict machinery (spec US4).

Slices: train (entry ts < OOS_BOUNDARY) vs OOS (entry ts >= OOS_BOUNDARY),
fixed at spec time (FR-007). Verdict compares candidate vs baseline on the
OOS slice only, one-sided bootstrap of the mean-R difference (10,000 draws,
seed 42 — reuses the same seeded-bootstrap discipline as edge_monitor).

Decision rule (pre-registered, spec.md SC-003 + edge cases):
  GO           iff n_oos >= 150 AND P(ΔavgR > 0) > 0.95 AND PF_cand >= PF_base
  KILL         when the OOS evidence is decisively against (p_win <= 0.5, i.e.
               the candidate is not even directionally better, or PF is worse)
  INCONCLUSIVE otherwise (30 <= n < 150 and directionally better but
               insignificant, or n < 30 — too thin to judge either way)
KILL is decisive from either slice; GO requires N>=150 (no thin-slice GO).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from selection_validator.dataset import OOS_BOUNDARY

DRAWS = 10_000
SEED = 42
N_REQUIRED = 150
ALPHA = 0.05


def split_slices(df: pd.DataFrame, boundary: str = OOS_BOUNDARY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train = entry ts < boundary; oos = entry ts >= boundary (FR-007)."""
    b = pd.Timestamp(boundary, tz="UTC")
    train = df[df["ts"] < b]
    oos = df[df["ts"] >= b]
    return train, oos


def bootstrap_diff(base_rs, cand_rs, draws: int = DRAWS, seed: int = SEED) -> float:
    """P(mean(cand) - mean(base) < 0) via paired-free resampling of both arrays."""
    base = np.asarray([float(r) for r in base_rs if r is not None], dtype=float)
    cand = np.asarray([float(r) for r in cand_rs if r is not None], dtype=float)
    if base.size < 2 or cand.size < 2:
        return 0.5
    rng = np.random.default_rng(seed)
    bm = rng.choice(base, size=(draws, base.size), replace=True).mean(axis=1)
    cm = rng.choice(cand, size=(draws, cand.size), replace=True).mean(axis=1)
    return float((cm - bm < 0).mean())


def decide(base: dict, cand: dict, n_oos: int) -> tuple[str, dict]:
    """Pre-registered GO / KILL / INCONCLUSIVE (spec.md SC-003 + edge cases)."""
    info = {"n_oos": n_oos, "n_cand": cand.get("closed", 0), "n_base": base.get("closed", 0)}
    if cand.get("closed", 0) < 30:
        return "INCONCLUSIVE", {**info, "why": "candidate slice too thin (<30 closed)"}
    if n_oos < N_REQUIRED:
        # No thin-slice GO. KILL only if the evidence is decisively against;
        # otherwise wait for more data (pre-registered).
        p_win = 1.0 - cand.get("p_delta_lt0", 0.5)
        if p_win <= 0.5 or cand.get("pf", 0) < base.get("pf", 0):
            return "KILL", {**info, "why": "decisively against on thin OOS"}
        return "INCONCLUSIVE", {**info, "why": f"n_oos={n_oos} < {N_REQUIRED} — waiting for more trades"}
    p_win = 1.0 - cand.get("p_delta_lt0", 0.5)
    if p_win > (1 - ALPHA) and cand.get("pf", 0) >= base.get("pf", 0):
        return "GO", {**info, "why": f"P(ΔavgR>0)={p_win:.3f} > 0.95, PF {cand.get('pf')} >= {base.get('pf')}"}
    if p_win <= 0.5 or cand.get("pf", 0) < base.get("pf", 0):
        return "KILL", {**info, "why": f"candidate not better (P(ΔavgR>0)={p_win:.3f}, PF {cand.get('pf')} vs {base.get('pf')})"}
    return "INCONCLUSIVE", {**info, "why": f"directionally better but not significant (P={p_win:.3f}) at n={n_oos}"}
