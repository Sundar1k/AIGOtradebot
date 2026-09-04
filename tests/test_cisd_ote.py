"""CISD+OTE strategy wiring. The detection/feature math is a verbatim port of the
training pipeline (so live == training); these check the integration — registered +
timeframe-gated, the 87-dim hand feature vector (76 FFM + 11 OTE/HTF geom), and that
detect() runs the zone pipeline without error. (grade() needs Chronos + the joblib,
so it's exercised by the backtest, not the fast suite.)"""
import numpy as np
import pandas as pd

import config
import strategies
from strategies.cisd_ote import CisdOteStrategy
from strategies import cisd_ote_detect as cod


def _bars(n=200):
    rng = np.arange(n)
    close = 100 + np.sin(rng / 7.0) * 5 + rng * 0.01
    return pd.DataFrame({
        "time": pd.date_range("2026-01-02 00:00", periods=n, freq="3min", tz="UTC"),
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": [1000] * n,
    })


def test_registered_and_timeframe_gated():
    assert "cisd_ote" in strategies.REGISTRY
    config.TIMEFRAME_MIN = 3
    assert CisdOteStrategy().has_model() is True            # cisd_ote_chronos.joblib ships
    config.TIMEFRAME_MIN = 5
    assert CisdOteStrategy().has_model() is False           # no 5-min variant → unavailable


def test_constants_match_training():
    # frozen params + derived TFs must match the trained model (no drift)
    assert cod.CISD_TF == 12 and cod.HTF_TF == "1h" and cod.N_GEOM == 11
    assert cod.PARAMS["entry_mode"] == "bot" and cod.PARAMS["sl_mode"] == "pivot"


def test_hand_features_is_87_dims():
    bars = _bars()
    s = CisdOteStrategy()
    sb = 150
    s._ctx = {
        "rec": {"is_long": True, "fib_top": 105.0, "fib_bot": 103.0, "zone_age": 4.0,
                "had_sweep": True, "disp": 1.2, "entry": 104.0, "sl": 102.0, "sig_bar": sb},
        "atr": np.full(len(bars), 2.0), "hour": np.full(len(bars), 10),
        "htf_sign": np.ones(len(bars)), "htf_str": np.full(len(bars), 0.01),
    }
    hand = s._hand_features(bars, sb, 1)
    assert hand.shape == (87,)                              # 76 FFM + 11 geom
    assert np.isfinite(hand).all()                          # NA-safe
    # geom block (last 11) sanity: direction is +1 for a long
    assert hand[76] == 1.0


def test_detect_runs_and_is_causal():
    # on benign synthetic bars it may or may not fire; either way it must not raise,
    # and any signal must reference a PAST bar (no look-ahead) with positive risk
    s = CisdOteStrategy()
    bars = _bars(300)
    sig = s.detect(bars)
    assert sig is None or (sig.bar_index < len(bars) - 1 and sig.risk > 0)


def test_detect_zone_signals_columns():
    # the verbatim detector returns the zone frame augmented with the expected cols
    bars = _bars(240)
    df3 = bars.set_index(pd.DatetimeIndex(bars["time"]))[["open", "high", "low", "close", "volume"]]
    zone = cod.resample(df3, cod.CISD_TF)
    out = cod.detect_zone_signals(zone, cod.PARAMS)
    for col in ("cisd_signal", "fib_top", "fib_bot", "origin", "had_sweep", "disp_strength"):
        assert col in out.columns
    assert set(np.unique(out["cisd_signal"])).issubset({0, 1, 2})
