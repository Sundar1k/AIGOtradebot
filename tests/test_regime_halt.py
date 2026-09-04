"""Regime-halt tests (spec T005, T012): state machine + verdict rule."""
import numpy as np
import pandas as pd
import pytest

from selection_validator.regime_halt import (HaltSimulator, decide_verdict,
                                             compare, build_take_rows)


def _row(ts, r, sym="NQ"):
    return {"ts": pd.Timestamp(ts, tz="UTC"), "symbol": sym, "strategy": "ema",
            "direction": 1, "proba": 0.4, "r_hat": 0.0, "entry": 100.0,
            "stop": 99.0, "risk": 1.0, "floor": 0.35, "ceil": 0.5,
            "take": True, "jump": False, "veto_quality": 6, "outcome_r": r}


def _df(rows):
    return pd.DataFrame(rows)


def test_cold_start_takes_everything():
    sim = HaltSimulator(_df([_row(f"2026-01-01 00:{i:02d}", -1.0) for i in range(10)]))
    dec, cand = sim.run()
    assert dec["go"].all()
    assert len(cand) == 10


def test_streak_trips_halt():
    rows = [_row(f"2026-01-01 00:{i:02d}", -1.0) for i in range(20)] + \
           [_row("2026-01-01 02:00", 0.5)]
    sim = HaltSimulator(_df(rows))
    dec, cand = sim.run()
    gos = dec["go"].tolist()
    # first 15 cold-start trades taken; the 16th (n=15 all losers) halts; the
    # rest of the streak + the final signal stay halted (cooldown)
    assert gos[:15] == [True] * 15
    assert all(not g for g in gos[15:])
    assert bool(dec.iloc[15]["edge_halt"])
    assert bool(dec.iloc[-1]["edge_halt"])


def test_cooldown_then_resume():
    rows = [_row(f"2026-01-01 00:{i:02d}", -1.0) for i in range(20)]       # halt at 16th
    rows += [_row("2026-01-01 03:00", 0.5),                                # +2h — still halted
             _row("2026-01-02 02:00", 0.5),                                # +26h — resume
             _row("2026-01-02 03:00", 0.5)]
    sim = HaltSimulator(_df(rows))
    dec, cand = sim.run()
    gos = dec["go"].tolist()
    assert gos[20] is False          # T+2h within cooldown
    assert gos[21] is True           # T+26h — cooldown elapsed, fresh window
    assert gos[22] is True           # still in fresh-window cold start

def test_chop_blocks_signal():
    rows = [_row("2026-01-01 00:00", 1.0), _row("2026-01-01 00:03", 1.0)]
    sim = HaltSimulator(_df(rows))
    # force the chop verdict for the second signal (integration path)
    sim._chop_cache[("NQ", str(rows[1]["ts"]))] = True
    dec, cand = sim.run()
    assert not bool(dec.iloc[1]["go"])
    assert bool(dec.iloc[1]["chop_halt"])


def test_verdict_go():
    comp = {
        "full": {"baseline": {"closed": 300, "avg_r": 0.5, "pf": 2.0},
                 "candidate": {"closed": 200, "avg_r": 0.8, "pf": 2.5},
                 "p_delta_gt0": 0.99},
        "august": {"baseline": {"avg_r": -1.2}, "candidate": {"avg_r": -0.3}},
    }
    v, checks = decide_verdict(comp)
    assert v == "GO" and all(checks.values())


def test_verdict_kill_when_august_not_improved():
    comp = {
        "full": {"baseline": {"closed": 300, "avg_r": 0.5, "pf": 2.0},
                 "candidate": {"closed": 200, "avg_r": 0.8, "pf": 2.5},
                 "p_delta_gt0": 0.99},
        "august": {"baseline": {"avg_r": -1.2}, "candidate": {"avg_r": -1.3}},
    }
    v, checks = decide_verdict(comp)
    assert v == "KILL" and checks["c_aug_improved"] is False


def test_verdict_kill_when_halts_everything():
    comp = {
        "full": {"baseline": {"closed": 300, "avg_r": 0.5, "pf": 2.0},
                 "candidate": {"closed": 30, "avg_r": 0.9, "pf": 3.0},
                 "p_delta_gt0": 0.99},
        "august": {"baseline": {"avg_r": -1.2}, "candidate": {"avg_r": -0.1}},
    }
    v, checks = decide_verdict(comp)
    assert v == "KILL" and checks["d_n_cand_ge_half"] is False


def test_verdict_inconclusive_thin():
    # thin but not halting-everything: n_cand >= half, directionally better,
    # insignificant -> INCONCLUSIVE
    comp = {
        "full": {"baseline": {"closed": 300, "avg_r": 0.5, "pf": 2.0},
                 "candidate": {"closed": 170, "avg_r": 0.55, "pf": 2.0},
                 "p_delta_gt0": 0.80},
        "august": {"baseline": {"avg_r": -1.2}, "candidate": {"avg_r": -0.9}},
    }
    v, _ = decide_verdict(comp)
    assert v == "INCONCLUSIVE"
