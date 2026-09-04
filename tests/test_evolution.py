"""Evolution tests (spec T004, T008, T011): evaluator, folds, verdict."""
import pandas as pd

from selection_validator.evolve_search import (candidate_mask, decide_verdict,
                                               make_folds)


def _df(n=30, start="2022-01-03 14:30:00"):
    # Jan 3 2022 is a Monday; 14:30 UTC = 09:30 EST; n=30 x 5min = exactly
    # the 150-minute window that candidate_mask always enforces.
    rows = []
    for i in range(n):
        rows.append({
            "ts": pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=5 * i),
            "symbol": "NQ", "strategy": "ema", "direction": 1,
            "proba": 0.4, "r_hat": 0.0, "entry": 100.0, "stop": 99.0,
            "risk": 1.0, "floor": 0.35, "ceil": 0.5, "take": True,
            "jump": False, "veto_quality": 6, "outcome_r": 0.5,
            "atr_ratio": 0.9,
        })
    return pd.DataFrame(rows)


def test_candidate_mask_floor_ceil():
    df = _df()
    df.loc[0, "proba"] = 0.20        # below floor 0.35
    df.loc[1, "proba"] = 0.55        # above ceil 0.50
    m = candidate_mask(df, 0.35, 0.50, None)
    assert not m.iloc[0] and not m.iloc[1]
    assert m.sum() == len(df) - 2


def test_chop_threshold_derived_from_ratio():
    df = _df()
    df["atr_ratio"] = [1.1] * len(df)          # above chop 1.0
    m10 = candidate_mask(df, 0.35, 0.50, 1.0)  # blocked
    m20 = candidate_mask(df, 0.35, 0.50, 2.0)  # allowed
    assert m10.sum() == 0 and m20.sum() == len(df)
    df["atr_ratio"] = [float("nan")] * len(df)  # fail-open
    assert candidate_mask(df, 0.35, 0.50, 1.0).sum() == len(df)


def test_folds_contiguous_no_overlap():
    df = _df(2000, "2021-04-01 14:30:00")   # ~7 days of window bars
    folds = make_folds(df)
    assert len(folds) >= 2
    seen = []
    for _, fd in folds:
        seen.extend(fd["ts"].tolist())
    assert len(set(seen)) == len(seen)


def test_verdict_inconclusive_thin():
    cand = {"n": 140, "pf": 1.8, "avg_r": 0.4}
    live = {"n": 39, "pf": 0.5, "avg_r": -0.5}
    aug_c = {"avg_r": -0.8}
    aug_l = {"avg_r": -1.2}
    v, checks = decide_verdict(cand, live, 200, aug_c, aug_l, 0.99)
    assert v == "INCONCLUSIVE" and not checks["d_n_cand_ge_150"]


def test_verdict_go_all_bars():
    cand = {"n": 200, "pf": 1.8, "avg_r": 0.4}
    live = {"n": 39, "pf": 0.5, "avg_r": -0.5}
    v, checks = decide_verdict(cand, live, 300, {"avg_r": -0.8}, {"avg_r": -1.2}, 0.99)
    assert v == "GO" and all(checks.values())


def test_verdict_kill_pf_worse():
    cand = {"n": 200, "pf": 0.4, "avg_r": 0.1}
    live = {"n": 39, "pf": 0.5, "avg_r": -0.5}
    v, checks = decide_verdict(cand, live, 300, {"avg_r": -0.9}, {"avg_r": -1.2}, 0.60)
    assert v == "KILL"
