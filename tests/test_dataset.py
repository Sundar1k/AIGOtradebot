"""Dataset tests (spec T005): schema round-trip, sink logging, trade matching."""
import json
import os
import tempfile

import pandas as pd
import pytest

from selection_validator.dataset import (ScoredSignal, SignalSink, leak_check,
                                         merge_outcomes)


class _Sig:
    name = "ema"
    direction = 1
    proba = 0.42
    r_hat = 0.3
    entry = 100.0
    stop = 99.5
    risk = 0.5


class _Bars:
    def __init__(self, t):
        self.df = pd.DataFrame({"time": [pd.Timestamp(t, tz="UTC")]})

    def __getitem__(self, k):
        if k == "time":
            return self.df["time"]
        raise KeyError(k)

    def __len__(self):
        return 1


def _fake_bars():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:03"], utc=True),
        "open": [100, 101], "high": [102, 103], "low": [99, 100],
        "close": [101, 102], "volume": [10, 12],
    })
    return df


def test_roundtrip():
    s = ScoredSignal(ts="2026-01-01 00:00:00+00:00", symbol="NQ", strategy="ema",
                     direction=1, proba=0.42, r_hat=0.3, entry=100.0, stop=99.5,
                     risk=0.5, floor=0.35, ceil=0.5, take=True)
    row = s.to_row()
    s2 = ScoredSignal.from_row(row)
    assert s2.proba == 0.42 and s2.take is True


def test_sink_writes_and_merge_outcomes():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "signals.jsonl")
        sink = SignalSink(path, "NQ")
        sig = _Sig()
        sink(sig, sig, 0.35, 0.50, _fake_bars())
        sink.close()
        rows = [json.loads(l) for l in open(path) if l.strip()]
        assert len(rows) == 1
        assert rows[0]["take"] is True
        assert rows[0]["outcome_r"] is None

        # merge a closed trade by (strategy, direction, entry, entry bar time)
        t = type("T", (), {"strategy": "ema", "direction": 1, "entry": 100.0,
                           "entry_time": pd.Timestamp("2026-01-01 00:03", tz="UTC"),
                           "r": 1.5, "reason": "target"})()
        matched = merge_outcomes(path, [t])
        assert matched == 1
        rows = [json.loads(l) for l in open(path) if l.strip()]
        assert rows[0]["outcome_r"] == 1.5
        assert rows[0]["exit_reason"] == "target"


def test_leak_check_clean():
    df = pd.DataFrame([{
        "ts": pd.Timestamp("2026-01-01", tz="UTC"), "proba": 0.4,
        "take": True, "jump": False, "outcome_r": 1.0,
    }])
    assert leak_check(df) == []


def test_leak_check_catches_missing_proba():
    df = pd.DataFrame([{
        "ts": pd.Timestamp("2026-01-01", tz="UTC"), "proba": None,
        "take": True, "jump": False, "outcome_r": 1.0,
    }])
    assert leak_check(df) != []
