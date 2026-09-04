"""Mechanics-audit tests (spec T003, T006, T008): census + removal rule."""
import pandas as pd

from selection_validator.mechanics_audit import (LIVE_FLOOR, LIVE_CEIL, census,
                                                 removal_flags, lane_audit)


def _df():
    rows = []
    ts0 = pd.Timestamp("2026-03-02 14:30:00", tz="UTC")   # 09:30 EST, in window
    for i in range(6):                                     # 6 bars x 2 strategies
        ts = ts0 + pd.Timedelta(minutes=5 * i)
        for lane in ("ema", "orb"):
            rows.append({
                "ts": ts, "symbol": "NQ", "strategy": lane,
                "direction": 1, "proba": 0.5, "r_hat": 0.3, "entry": 100.0,
                "stop": 99.0, "risk": 1.0, "floor": 0.35, "ceil": 0.5,
                "take": True, "jump": False, "veto_quality": 6,
                "outcome_r": 0.5, "atr_ratio": 0.9,
            })
    return pd.DataFrame(rows)


def test_census_dual_fire_detected():
    df = _df()
    df = df.assign(eligible=((df["proba"] >= LIVE_FLOOR) & (df["proba"] <= LIVE_CEIL)))
    c = census(df)
    assert c["dual_fire_events"] == 6          # 6 ts x 2 strategies
    assert c["direction_agreement_rate"] == 1.0


def test_removal_flag_negative_lane():
    # 40 ema trades (all losers) + 40 orb trades (all winners), all in the
    # hold-out slice -> ema must be flagged (n>=30, negative, P>0.95 worse)
    rows = []
    ts0 = pd.Timestamp("2026-03-02 14:30:00", tz="UTC")
    for i in range(80):
        rows.append({
            "ts": ts0 + pd.Timedelta(minutes=5 * i),
            "symbol": "NQ", "strategy": "ema" if i < 40 else "orb",
            "direction": 1, "proba": 0.5, "r_hat": 0.0, "entry": 100.0,
            "stop": 99.0, "risk": 1.0, "floor": 0.35, "ceil": 0.5,
            "take": True, "jump": False, "veto_quality": 6,
            "outcome_r": -0.8 if i < 40 else 0.6, "atr_ratio": 0.9,
        })
    df = pd.DataFrame(rows)
    df = df.assign(eligible=True)
    flags = removal_flags(lane_audit(df), df)
    assert bool(flags.loc[flags["lane"] == "ema", "flag"].iloc[0])


def test_removal_no_flag_when_positive():
    df = _df()
    df["outcome_r"] = 0.5
    df = df.assign(eligible=True)
    flags = removal_flags(lane_audit(df), df)
    assert not flags["flag"].any()
