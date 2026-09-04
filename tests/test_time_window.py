"""Time-window tests (spec T003, T006): boundaries + verdict rule."""
import pandas as pd

from selection_validator.time_window import et_minute, window_mask, decide_verdict


def _df():
    # March 1 2026 = EST (DST starts Mar 8 2026) -> 13:30 UTC = 08:30 ET
    # July 15 2026 = EDT -> 13:30 UTC = 09:30 ET  (same UTC, different ET!)
    return pd.DataFrame({
        "ts": pd.to_datetime([
            "2026-03-01 13:29:59",   # 08:29:59 ET (EST) — outside
            "2026-03-01 13:30:00",   # 08:30:00 ET (EST) — outside
            "2026-03-01 14:29:59",   # 09:29:59 ET (EST) — outside
            "2026-03-01 14:30:00",   # 09:30:00 ET (EST) — inside
            "2026-03-01 17:00:00",   # 12:00:00 ET (EST) — outside
            "2026-07-15 13:30:00",   # 09:30:00 ET (EDT) — inside
            "2026-07-15 16:00:00",   # 12:00:00 ET (EDT) — outside
        ], utc=True),
        "outcome_r": [1.0] * 7,
        "take": [True] * 7, "jump": [False] * 7,
    })


def test_et_minute_dst_correct():
    m = et_minute(_df()["ts"])
    # EST (UTC-5): 13:30 UTC -> 08:30 ET
    assert list(m)[:2] == [8 * 60 + 29, 8 * 60 + 30]
    # same UTC clock in EDT (UTC-4): 13:30 UTC -> 09:30 ET
    assert list(m)[5] == 9 * 60 + 30
    assert list(m)[4] == 12 * 60          # 17:00 UTC EST -> 12:00 ET


def test_window_boundaries():
    w = window_mask(_df())
    # in-window: 14:30-17:00 UTC in EST (09:30-12:00 ET) + 13:30-16:00 EDT
    assert list(w) == [False, False, False, True, False, True, False]


def test_verdict_go():
    comp = {
        "oos": {"baseline": {"closed": 400, "avg_r": 0.5, "pf": 2.0},
                "window": {"closed": 180, "avg_r": 0.8, "pf": 2.6},
                "p_delta_gt0": 0.99},
        "august": {"baseline": {"avg_r": -1.2}, "window": {"avg_r": -0.8}},
    }
    v, checks = decide_verdict(comp)
    assert v == "GO" and all(checks.values())


def test_verdict_kill_august_worse():
    comp = {
        "oos": {"baseline": {"closed": 400, "avg_r": 0.5, "pf": 2.0},
                "window": {"closed": 180, "avg_r": 0.8, "pf": 2.6},
                "p_delta_gt0": 0.99},
        "august": {"baseline": {"avg_r": -1.2}, "window": {"avg_r": -1.4}},
    }
    v, checks = decide_verdict(comp)
    assert v == "KILL" and not checks["c_aug_not_worse"]


def test_verdict_inconclusive_thin():
    comp = {
        "oos": {"baseline": {"closed": 400, "avg_r": 0.5, "pf": 2.0},
                "window": {"closed": 60, "avg_r": 0.55, "pf": 2.1},
                "p_delta_gt0": 0.80},
        "august": {"baseline": {"avg_r": -1.2}, "window": {"avg_r": -0.9}},
    }
    v, _ = decide_verdict(comp)
    assert v == "INCONCLUSIVE"
