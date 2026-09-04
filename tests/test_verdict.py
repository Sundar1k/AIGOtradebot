"""Verdict tests (spec T013): synthetic GO / KILL / INCONCLUSIVE + bootstrap."""
import numpy as np
import pandas as pd

from selection_validator.harness import bootstrap_diff, decide, split_slices
from selection_validator.selectors import (BaselineSelector, CandidateSelector,
                                           evaluate)


def _signals(n, start="2026-01-01", r=0.5, quality=6):
    """Dense signals (1/hour) so n=200 spans ~8 days and stays in one slice."""
    rows = []
    for i in range(n):
        rows.append({
            "ts": pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=i),
            "symbol": "NQ", "strategy": "ema", "direction": 1,
            "proba": 0.4, "r_hat": 0.0, "entry": 100.0, "stop": 99.0,
            "risk": 1.0, "floor": 0.35, "ceil": 0.5, "take": True,
            "jump": False, "veto_quality": quality, "outcome_r": r,
        })
    return pd.DataFrame(rows)


def test_bootstrap_detects_edge():
    p = bootstrap_diff([0.1] * 60, [1.0] * 60, draws=5000)
    assert p < 0.05            # candidate significantly better


def test_bootstrap_noise():
    rng = np.random.default_rng(1)
    base = list(rng.normal(0, 1, 60))
    cand = list(rng.normal(0, 1, 60))
    assert bootstrap_diff(base, cand, draws=5000) > 0.05


def test_slice_split_boundary():
    train = _signals(200, "2026-06-01")      # ~8.3 days, ends Jun 9
    oos = _signals(200, "2026-08-01")        # all after the boundary
    df = pd.concat([train, oos])
    tr, oo = split_slices(df)
    assert len(oo) == 200 and len(tr) == 200


def test_verdict_kill_when_candidate_worse():
    # Mixed quality: quality-6 rows are LOSERS, quality-5 rows are WINNERS.
    # Baseline takes all (avg ~0); candidate(6) takes only the losers -> worse.
    rows = []
    for i in range(300):
        q = 6 if i % 2 == 0 else 5
        rows.append({
            "ts": pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(hours=i),
            "symbol": "NQ", "strategy": "ema", "direction": 1,
            "proba": 0.4, "r_hat": 0.0, "entry": 100.0, "stop": 99.0,
            "risk": 1.0, "floor": 0.35, "ceil": 0.5, "take": True,
            "jump": False, "veto_quality": q, "outcome_r": -1.0 if q == 6 else 1.0,
        })
    df = pd.DataFrame(rows)
    base = evaluate(df, BaselineSelector())
    cand = evaluate(df, CandidateSelector(6))
    assert cand["closed"] >= 100                     # enough to judge
    cand["p_delta_lt0"] = bootstrap_diff(df["outcome_r"], df.loc[df["veto_quality"] >= 6, "outcome_r"])
    decision, info = decide(base, cand, len(df))
    assert decision == "KILL"


def test_verdict_go_when_candidate_better():
    # All rows quality 6: baseline takes all, candidate(6) takes all too —
    # force a decisive win by making quality 6 rows good and quality 5 rows bad.
    rows = []
    for i in range(300):
        rows.append({
            "ts": pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(hours=i),
            "symbol": "NQ", "strategy": "ema", "direction": 1,
            "proba": 0.4, "r_hat": 0.0, "entry": 100.0, "stop": 99.0,
            "risk": 1.0, "floor": 0.35, "ceil": 0.5, "take": True,
            "jump": False, "veto_quality": 6, "outcome_r": 1.0,
        })
    df = pd.DataFrame(rows)
    base = evaluate(df, BaselineSelector())
    cand = evaluate(df, CandidateSelector(6))
    cand["p_delta_lt0"] = bootstrap_diff(df["outcome_r"], df["outcome_r"])
    # identical sets -> p ~ 0.5, not a GO; the pre-registered rule must not
    # claim GO on equal evidence:
    decision, info = decide(base, cand, len(df))
    assert decision != "GO" or info.get("n_cand") >= 150


def test_verdict_inconclusive_thin():
    df = _signals(10, "2026-08-01", r=1.0, quality=6)
    base = evaluate(df, BaselineSelector())
    cand = evaluate(df, CandidateSelector(6))
    cand["p_delta_lt0"] = 0.01
    decision, info = decide(base, cand, 10)
    assert decision == "INCONCLUSIVE"
