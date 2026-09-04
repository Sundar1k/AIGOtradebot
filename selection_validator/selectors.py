"""selectors.py — baseline and candidate selectors (FR-002: shared evaluator).

BaselineSelector  = the current funnel: take rows where floor <= proba <= ceil
                    AND not jump-skipped (the live gate stack is recorded in
                    each row's gate outcomes; replay applies the deterministic
                    subset — jump — while live gates run live).
CandidateSelector = baseline funnel AND veto_quality >= q (q in {5,6,7}).

Both share one evaluate() code path (FR-002). Stats are computed on the
taken rows' realized outcome_r (NET of slippage + commission as recorded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BaselineSelector:
    name = "baseline"

    def accept(self, df: pd.DataFrame) -> pd.Series:
        # take already encodes floor<=proba<=ceil; jump skip is a selection
        # filter of the baseline funnel.
        return df["take"].astype(bool) & ~df["jump"].astype(bool)


class CandidateSelector:
    """Baseline funnel + veto-quality band (doctrine sanctioned lever #1)."""

    def __init__(self, q: int):
        self.q = int(q)
        self.name = f"quality>=q"

    def accept(self, df: pd.DataFrame) -> pd.Series:
        base = df["take"].astype(bool) & ~df["jump"].astype(bool)
        if "veto_quality" in df.columns:
            qual = df["veto_quality"].fillna(0).astype(int)
        else:
            qual = pd.Series(0, index=df.index)
        return base & (qual >= self.q)


def stats(rs) -> dict:
    rs = np.asarray([float(r) for r in rs if r is not None and not np.isnan(r)])
    n = int(rs.size)
    if n == 0:
        return {"n": 0, "wr": None, "avg_r": None, "sum_r": None, "pf": None}
    wins = rs[rs > 0].sum()
    losses = -rs[rs < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    return {
        "n": n,
        "wr": round(float((rs > 0).mean()), 4),
        "avg_r": round(float(rs.mean()), 4),
        "sum_r": round(float(rs.sum()), 4),
        "pf": pf,
    }


def evaluate(df: pd.DataFrame, selector) -> dict:
    """Shared evaluator (FR-002): one code path for baseline and candidates.

    Universe = rows the selector accepts AND that have a realized outcome
    (unclosed trades cannot contribute R yet).
    """
    mask = selector.accept(df)
    rs = df.loc[mask, "outcome_r"]
    out = stats(rs)
    out["selector"] = selector.name
    out["signals"] = int(mask.sum())
    out["closed"] = int((mask & df["outcome_r"].notna()).sum())
    return out
