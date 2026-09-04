#!/usr/bin/env python3
"""colabs/labelers.py — concrete Chronos labelers for the ACTIVE lanes.

Each labeler reimplements the signal function exactly as lane_test.py does
(the sanctioned comparison harness, itself matching strategies/*.py) and the
handcrafts exactly as the strategy's _hand_features (minus the 76 FFM cols,
which BaseLabeler.features() adds). Handcraft counts match the shipped
bundles: ema=5 (feat_dim 337), supertrend=2 (334), orb=7 (339); gann mirrors
supertrend (2) — the strategy docstring says its hand features mirror
supertrend (adx + ffm).

tf is a class attr so `labeler = EMACrossChronos(tf="15min")` trains a
15-min variant through the SAME walk-forward pipeline.
"""
import numpy as np

import config
import indicators as ind
from colabs.base_labeler import BaseLabeler, CTX


class EMACrossChronos(BaseLabeler):
    """9/20 EMA cross, ADX-gated (the ema lane). 5 handcrafts."""

    def __init__(self, tf="3min"):
        self.tf = tf
        super().__init__()

    def _fired(self, key, df, i):
        fc = getattr(df, "_ff_cache")
        c, ef, es, adx = fc["c"], fc["ef"], fc["es"], fc["adx"]
        if not (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])
                and np.isfinite(adx[i])):
            return None
        if config.ADX_GATE and adx[i] < config.ADX_GATE:
            return None
        if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
            return 1
        if ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
            return -1
        return None

    def _hand(self, key, df, i, d):
        fc = getattr(df, "_ff_cache")
        c, ef, es = fc["c"], fc["ef"], fc["es"]
        atr_i = fc["atr"][i]
        a = atr_i if (np.isfinite(atr_i) and atr_i > 0) else np.nan
        def g(x):
            return float(x) if np.isfinite(x) else 0.0
        with np.errstate(invalid="ignore"):
            ema_spread = (ef[i] - es[i]) / a * d
            k = config.SLOW_SLOPE_K
            slow_slope = ((es[i] - es[i - k]) / a * d) if i - k >= 0 else np.nan
            price_vs_slow = (c[i] - es[i]) / a * d
        adx_i, adx_slope = fc["adx"][i], (fc["adx"][i] - fc["adx"][i - config.ADX_SLOPE]) if i >= config.ADX_SLOPE else 0.0
        return np.asarray([g(ema_spread), g(slow_slope), g(price_vs_slow),
                           adx_i / 100.0, adx_slope / 100.0], np.float32)

    def handcraft_names(self):
        return ["ema_spread", "slow_slope", "price_vs_slow",
                "adx", "adx_slope"]


class SuperTrendChronos(BaseLabeler):
    """SuperTrend flip, ADX not gated (the st lane). 2 handcrafts."""

    def __init__(self, tf="3min"):
        self.tf = tf
        super().__init__()

    def _scan(self, key, df):
        """Precompute supertrend once per df, then scan."""
        if not hasattr(df, "_st_dir"):
            st_line, st_dir = ind.supertrend(df, config.ST_PERIOD, config.ST_MULT)
            df._st_dir = np.asarray(st_dir, dtype=float)
        return super()._scan(key, df)

    def _fired(self, key, df, i):
        st = getattr(df, "_st_dir")
        if i < 1 or not (np.isfinite(st[i]) and np.isfinite(st[i - 1])):
            return None
        if st[i] == st[i - 1]:
            return None
        return int(st[i])

    def _hand(self, key, df, i, d):
        fc = getattr(df, "_ff_cache")
        adx_i = fc["adx"][i] if np.isfinite(fc["adx"][i]) else 0.0
        slope = (fc["adx"][i] - fc["adx"][i - config.ADX_SLOPE]) if i >= config.ADX_SLOPE else 0.0
        return np.asarray([float(adx_i), float(slope)], np.float32)

    def handcraft_names(self):
        return ["adx", "adx_slope"]


class ORBChronos(BaseLabeler):
    """Opening-range breakout, time + ADX gated (the orb lane). 7 handcrafts."""

    def __init__(self, tf="3min"):
        self.tf = tf
        super().__init__()

    def _scan(self, key, df):
        if not hasattr(df, "_orb"):
            oh, ol = ind.opening_range(df, config.ORB_BARS, config.ORB_OPEN_MIN,
                                       config.ORB_TZ)
            df._orb = (np.asarray(oh, dtype=float), np.asarray(ol, dtype=float))
            df._et_min = np.asarray(ind.et_minutes(df, config.ORB_TZ), dtype=float)
        return super()._scan(key, df)

    def _fired(self, key, df, i):
        fc = getattr(df, "_ff_cache")
        oh, ol = getattr(df, "_orb")
        et = getattr(df, "_et_min")
        c = fc["c"]
        if not (np.isfinite(oh[i]) and np.isfinite(oh[i - 1])
                and np.isfinite(ol[i]) and np.isfinite(ol[i - 1])
                and np.isfinite(fc["adx"][i])):
            return None
        if et[i] >= config.ORB_CLOSE_MIN or fc["adx"][i] < config.ORB_ADX_GATE:
            return None
        if c[i - 1] <= oh[i - 1] and c[i] > oh[i]:
            return 1
        if c[i - 1] >= ol[i - 1] and c[i] < ol[i]:
            return -1
        return None

    def _hand(self, key, df, i, d):
        fc = getattr(df, "_ff_cache")
        c, v = fc["c"], df["volume"].to_numpy(float)
        oh, ol = getattr(df, "_orb")
        sess_open, prior_close, or_avg_vol = ind.orb_extras(
            df, config.ORB_BARS, config.ORB_OPEN_MIN, config.ORB_TZ)
        atr_i = fc["atr"][i]
        a = atr_i if (np.isfinite(atr_i) and atr_i > 0) else np.nan
        def g(x):
            return float(x) if np.isfinite(x) else 0.0
        level = oh[i] if d == 1 else ol[i]
        or_mid = 0.5 * (oh[i] + ol[i])
        half = 0.5 * (oh[i] - ol[i])
        ov = or_avg_vol[i]
        with np.errstate(invalid="ignore"):
            or_size = (oh[i] - ol[i]) / a
            breakout_ext = (c[i] - level) / a * d
            session_gap = (sess_open[i] - prior_close[i]) / a * d
            approach_pos = (c[i - 1] - or_mid) / half * d
            or_vol_ratio = (v[i] / ov) if (np.isfinite(ov) and ov > 0) else np.nan
        adx_i = fc["adx"][i] if np.isfinite(fc["adx"][i]) else 0.0
        slope = (fc["adx"][i] - fc["adx"][i - config.ADX_SLOPE]) if i >= config.ADX_SLOPE else 0.0
        return np.asarray([g(or_size), g(breakout_ext), g(session_gap),
                           g(approach_pos), g(or_vol_ratio),
                           adx_i / 100.0, slope / 100.0], np.float32)

    def handcraft_names(self):
        return ["or_size", "breakout_ext", "session_gap", "approach_pos",
                "or_vol_ratio", "adx", "adx_slope"]


class GannChronos(BaseLabeler):
    """Gann 1x1 angle-line flip (the gann lane). 2 handcrafts (mirrors
    supertrend per the strategy docstring)."""

    def __init__(self, tf="3min"):
        self.tf = tf
        super().__init__()

    def _scan(self, key, df):
        """Precompute the fractal swing lows/highs ONCE per df (vectorized,
        ~1000x faster than the per-bar Python swing search) then scan."""
        if not hasattr(df, "_gann_swings"):
            from numpy.lib.stride_tricks import sliding_window_view
            lo = df["low"].to_numpy(float)
            hi = df["high"].to_numpy(float)
            n = len(lo)
            k = getattr(config, "GANN_SWING_K", 5)
            w = 2 * k + 1
            lo_pad = np.concatenate([np.full(k, lo[0]), lo, np.full(k, lo[-1])])
            hi_pad = np.concatenate([np.full(k, hi[0]), hi, np.full(k, hi[-1])])
            wlo = sliding_window_view(lo_pad, w)
            whi = sliding_window_view(hi_pad, w)
            lows = np.zeros(n, bool)
            highs = np.zeros(n, bool)
            for i in range(n):
                lows[i] = (lo[i] == wlo[i].min())
                highs[i] = (hi[i] == whi[i].max())
            df._gann_swings = (lows, highs)
        return super()._scan(key, df)

    def _fired(self, key, df, i):
        fc = getattr(df, "_ff_cache")
        c, lo, hi = fc["c"], fc["lo"], fc["hi"]
        atr = fc["atr"]
        if i < 2 * getattr(config, "GANN_SWING_K", 5) + 1:
            return None
        if not (np.isfinite(atr[i]) and np.isfinite(atr[i - 1]) and atr[i] > 0):
            return None
        k = getattr(config, "GANN_SWING_K", 5)
        m = getattr(config, "GANN_SLOPE_MULT", 0.5)
        lows, highs = getattr(df, "_gann_swings")
        # last confirmed swing low/high at or before bar i (fractal window)
        low_i, high_i = -1, -1
        for j in range(i - k - 1, k - 1, -1):
            if low_i < 0 and lows[j]:
                low_i = j
            if high_i < 0 and highs[j]:
                high_i = j
        if low_i < 0 or high_i < 0:
            return None
        up = lo[low_i] + m * atr[i] * (i - low_i)
        dn = hi[high_i] - m * atr[i] * (i - high_i)
        p_up = c[i - 1] > (lo[low_i] + m * atr[i - 1] * (i - 1 - low_i))
        p_dn = c[i - 1] < (hi[high_i] - m * atr[i - 1] * (i - 1 - high_i))
        c_up = c[i] > up
        c_dn = c[i] < dn
        if p_up and not c_up and not c_dn:
            return -1
        if p_dn and not c_dn and not c_up:
            return 1
        if not p_up and not p_dn:
            if c_up:
                return 1
            if c_dn:
                return -1
        return None

    def _hand(self, key, df, i, d):
        fc = getattr(df, "_ff_cache")
        adx_i = fc["adx"][i] if np.isfinite(fc["adx"][i]) else 0.0
        slope = (fc["adx"][i] - fc["adx"][i - config.ADX_SLOPE]) if i >= config.ADX_SLOPE else 0.0
        return np.asarray([float(adx_i), float(slope)], np.float32)

    def handcraft_names(self):
        return ["adx", "adx_slope"]
