#!/usr/bin/env python3
"""candle_patterns.py — live candlestick-pattern detection for the scanner.

Resamples the bot's 3-min bars to 30-min and detects classic single- and
two-bar patterns on the LAST few candles. Used by supervisor.py to log what
the chart is doing ("🕯 YM 30m: MARUBOZU BULL") and by missed_trades.py to
tag every signal with the pattern present at signal time, so the ledger can
validate whether any pattern actually predicts winners.

Patterns (classic definitions, no model):
  DOJI             body < 10% of range            (indecision)
  HAMMER           long lower wick, small body    (reversal up at lows)
  INVERTED HAMMER  long upper wick, small body    (weak, at lows)
  SHOOTING STAR    long upper wick, small body    (reversal down at highs)
  BULL/BEAR ENGULFING  two-bar: body covers prev  (momentum flip)
  MARUBOZU BULL/BEAR   body > 80% of range        (strong trend)
  INSIDE BAR       current range inside prev      (compression)

Design notes:
  - Pure numpy/pandas; no model, no training. Observation-first: patterns
    are LOGGED and TAGGED, never gating until the ledger proves an edge.
  - 30-min aggregation matches how a human reads the chart (the user asked
    for the 30-min view); the 3-min live candles feed it.
"""

import numpy as np
import pandas as pd

RESAMPLE = "30min"

# Directional meaning of each pattern (chart-pattern-detector methodology).
# +1 = bullish, -1 = bearish, 0 = neutral/indecision.
PATTERN_DIR = {
    "DOJI": 0,
    "INSIDE BAR": 0,
    "MARUBOZU BULL": 1,
    "MARUBOZU BEAR": -1,
    "HAMMER": 1,
    "INVERTED HAMMER": 1,          # weak bullish (at lows)
    "SHOOTING STAR": -1,
    "BULL ENGULFING": 1,
    "BEAR ENGULFING": -1,
    "HIGH-VOL BULL REVERSAL": 1,
    "HIGH-VOL BULL WICK": 1,
    "HIGH-VOL BEAR REVERSAL": -1,
    "HIGH-VOL BEAR WICK": -1,
}


def pattern_direction(pats: list) -> int:
    """Net directional vote of a pattern list: +1 bull, -1 bear, 0 neutral.
    Bull patterns cancel bear patterns one-for-one; ties -> 0."""
    if not pats:
        return 0
    score = sum(PATTERN_DIR.get(p, 0) for p in pats)
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def resample_30m(df: pd.DataFrame) -> pd.DataFrame:
    """3-min bars -> 30-min bars (open/high/low/close/volume)."""
    d = df.copy()
    if "time" not in d.columns:
        return d
    d = d.set_index("time")
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    out = d.resample(RESAMPLE).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return out.reset_index()


def detect_on_candle(o, h, l, c, prev_o=None, prev_c=None,
                     prev_h=None, prev_l=None, vol=None, vol_ma=None) -> list:
    """Pattern names for ONE 30-min candle (plus context candle for
    two-bar patterns). Returns a list of pattern strings.

    vol / vol_ma: current volume and its rolling mean (e.g. 20-bar) — used
    for volume-confirmed reversals (accumulation/distribution) per the
    chart-pattern-detector methodology: massive volume + wick rejection."""
    rng = h - l
    if rng <= 0:
        return []
    body = abs(c - o)
    up = c > o
    upper = h - max(o, c)
    lower = min(o, c) - l
    pats = []
    # volume surge: >= 2x the rolling mean (when volume data exists)
    vol_surge = (vol is not None and vol_ma is not None
                 and vol_ma > 0 and vol >= 2.0 * vol_ma)

    if body / rng < 0.10:
        pats.append("DOJI")
    if body > 0.80 * rng:
        pats.append("MARUBOZU BULL" if up else "MARUBOZU BEAR")

    # wick patterns (require a real body so a doji isn't double-counted)
    if body / rng >= 0.10:
        if lower >= 2.0 * body and upper <= 0.35 * body and lower > 0.45 * rng:
            pats.append("HAMMER" if not up else "INVERTED HAMMER")
        if upper >= 2.0 * body and lower <= 0.35 * body and upper > 0.45 * rng:
            pats.append("SHOOTING STAR" if up else "INVERTED HAMMER")

    # volume-confirmed wick rejections (accumulation / distribution)
    if vol_surge:
        if lower >= 1.5 * body and lower > 0.4 * rng:
            pats.append("HIGH-VOL BULL REVERSAL" if not up
                        else "HIGH-VOL BULL WICK")
        if upper >= 1.5 * body and upper > 0.4 * rng:
            pats.append("HIGH-VOL BEAR REVERSAL" if up
                        else "HIGH-VOL BEAR WICK")

    # two-bar patterns
    if prev_o is not None and prev_c is not None and prev_h and prev_l:
        if up and prev_c < prev_o and c >= prev_o and o <= prev_c:
            pats.append("BULL ENGULFING")
        if not up and prev_c > prev_o and c <= prev_o and o >= prev_c:
            pats.append("BEAR ENGULFING")
        if prev_h is not None and prev_l is not None:
            if h <= prev_h and l >= prev_l:
                pats.append("INSIDE BAR")
    return pats


def detect_patterns(df: pd.DataFrame, n_last: int = 3) -> list:
    """Detect patterns on the last `n_last` 30-min candles.
    Returns [(time_str, [pattern, ...]), ...] newest last."""
    d = resample_30m(df)
    if len(d) < 2:
        return []
    vol_ma = d["volume"].rolling(20, min_periods=5).mean() if "volume" in d else None
    out = []
    for i in range(max(0, len(d) - n_last), len(d)):
        r = d.iloc[i]
        v = float(r["volume"]) if vol_ma is not None and pd.notna(r["volume"]) else None
        vm = float(vol_ma.iloc[i]) if vol_ma is not None and pd.notna(vol_ma.iloc[i]) else None
        if i >= 1:
            p = d.iloc[i - 1]
            pats = detect_on_candle(r["open"], r["high"], r["low"], r["close"],
                                    p["open"], p["close"], p["high"], p["low"],
                                    vol=v, vol_ma=vm)
        else:
            pats = detect_on_candle(r["open"], r["high"], r["low"], r["close"],
                                    vol=v, vol_ma=vm)
        out.append((str(r["time"])[:16], pats))
    return out


def pattern_at_time(df: pd.DataFrame, ts) -> list:
    """Patterns on the 30-min candle CONTAINING timestamp `ts` (for tagging
    signals in the missed-trade ledger). Returns list of pattern names."""
    d = resample_30m(df)
    if len(d) < 2:
        return []
    vol_ma = d["volume"].rolling(20, min_periods=5).mean() if "volume" in d else None
    ts = pd.Timestamp(ts)
    idx = d.index[d["time"] <= ts]
    if len(idx) == 0:
        return []
    i = d.index.get_loc(idx[-1])
    r = d.iloc[i]
    v = float(r["volume"]) if vol_ma is not None and pd.notna(r["volume"]) else None
    vm = float(vol_ma.iloc[i]) if vol_ma is not None and pd.notna(vol_ma.iloc[i]) else None
    p = d.iloc[i - 1] if i >= 1 else None
    if p is not None:
        pats = detect_on_candle(r["open"], r["high"], r["low"], r["close"],
                                p["open"], p["close"], p["high"], p["low"],
                                vol=v, vol_ma=vm)
    else:
        pats = detect_on_candle(r["open"], r["high"], r["low"], r["close"],
                                vol=v, vol_ma=vm)
    return pats


if __name__ == "__main__":
    # quick self-test on synthetic data
    import datetime as dt
    idx = pd.date_range("2026-08-17 00:00", periods=200, freq="3min")
    rng = np.random.RandomState(1)
    base = 5000
    closes = [base]
    for _ in range(199):
        closes.append(closes[-1] + rng.randn() * 3)
    df = pd.DataFrame({
        "time": idx,
        "open": closes[:-1], "high": np.array(closes[1:]) + 2,
        "low": np.array(closes[1:]) - 2, "close": closes[1:],
        "volume": rng.randint(1, 100, 199),
    })
    for t, pats in detect_patterns(df):
        if pats:
            print(t, pats)
