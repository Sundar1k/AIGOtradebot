"""Selector tests (spec T008, T011): baseline sanity + q-threshold behavior."""
import pandas as pd

from selection_validator.selectors import (BaselineSelector, CandidateSelector,
                                           evaluate, stats)


def _df(rows):
    return pd.DataFrame(rows)


def test_baseline_accepts_take_not_jump():
    df = _df([
        {"take": True, "jump": False, "outcome_r": 1.0},
        {"take": True, "jump": True, "outcome_r": -1.0},
        {"take": False, "jump": False, "outcome_r": 2.0},
    ])
    acc = BaselineSelector().accept(df)
    assert list(acc) == [True, False, False]


def test_candidate_q_threshold():
    df = _df([
        {"take": True, "jump": False, "veto_quality": 6, "outcome_r": 1.0},
        {"take": True, "jump": False, "veto_quality": 5, "outcome_r": -1.0},
        {"take": True, "jump": False, "veto_quality": 0, "outcome_r": 0.5},  # unscored
    ])
    acc6 = CandidateSelector(6).accept(df)
    acc5 = CandidateSelector(5).accept(df)
    assert list(acc6) == [True, False, False]
    assert list(acc5) == [True, True, False]


def test_evaluate_stats():
    df = _df([
        {"take": True, "jump": False, "outcome_r": 2.0},
        {"take": True, "jump": False, "outcome_r": -1.0},
        {"take": True, "jump": False, "outcome_r": None},   # open — excluded
        {"take": True, "jump": False, "outcome_r": 1.0},
    ])
    out = evaluate(df, BaselineSelector())
    assert out["closed"] == 3
    assert abs(out["wr"] - 2 / 3) < 0.001
    assert abs(out["avg_r"] - 2 / 3) < 1e-3
    assert out["pf"] == 3.0


def test_stats_all_losses_pf_zero():
    out = stats([-1.0, -1.5])
    assert out["pf"] == 0.0
    assert out["n"] == 2
