"""ORB session time gate — no stale overnight breakouts.

The 09:30-ET opening range stays mathematically "active" until midnight ET, so an
ungated ORB fires breakouts of the morning range all evening (the −1R orb short at
21:45 ET). _fired must gate entries to the RTH window [~09:45, 16:00) ET."""
import pandas as pd

import config
from strategies.orb import OrbStrategy


def _orb_day(breakout_idx, n=260):
    """One ET session of 3-min bars from 09:30. First ORB_BARS bars define the
    range (high 100.5 / low 99.5); a single breakout (close 101 > range high) is
    injected at `breakout_idx`. Returns bars ending at that breakout bar."""
    t0 = pd.Timestamp("2026-06-15 09:30", tz=config.ORB_TZ)
    times = pd.date_range(t0, periods=n, freq="3min", tz=config.ORB_TZ)
    high = [100.5] * n
    low = [99.5] * n
    close = [100.0] * 5 + [99.8] * (n - 5)      # inside the range after it forms
    close[breakout_idx - 1] = 100.0             # prior bar at/below the high
    close[breakout_idx] = 101.0                 # breakout close above the high
    high[breakout_idx] = 101.2
    df = pd.DataFrame({"time": times, "open": close, "high": high, "low": low,
                       "close": close, "volume": [1000] * n})
    return df.iloc[: breakout_idx + 1]


def test_breakout_fires_during_rth():
    config.ORB_ADX_GATE = 0.0                   # isolate the time gate
    config.ORB_CLOSE_MIN = 16 * 60
    bars = _orb_day(breakout_idx=10)            # 09:30 + 10×3min = 10:00 ET
    assert OrbStrategy()._fired(bars) == 1


def test_breakout_gated_overnight():
    config.ORB_ADX_GATE = 0.0
    config.ORB_CLOSE_MIN = 16 * 60
    bars = _orb_day(breakout_idx=245)           # 09:30 + 245×3min = 21:45 ET
    assert bars["time"].iloc[-1].tz_convert(config.ORB_TZ).hour == 21
    assert OrbStrategy()._fired(bars) is None    # past 16:00 ET → gated


def test_gate_disabled_lets_overnight_through():
    config.ORB_ADX_GATE = 0.0
    config.ORB_CLOSE_MIN = 0                     # gate off → original behaviour
    bars = _orb_day(breakout_idx=245)
    assert OrbStrategy()._fired(bars) == 1
