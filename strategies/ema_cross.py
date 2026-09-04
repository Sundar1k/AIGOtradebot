#!/usr/bin/env python3
"""strategies/ema_cross.py — 9/20 EMA crossover strategy.

A fast/slow EMA crossover (gated by ADX ≥ ADX_GATE so it only fires in a
trending regime) is the mechanical entry; the EMACrossChronos model grades
whether the cross rides or whips. Hand features = 76 FFM +
[ema_spread, slow_slope, price_vs_slow, adx/100, adx_slope/100], each signed
by trade direction  (81 total → feat_dim 337).
"""
import numpy as np

import config
import indicators as ind
from strategies.base import Strategy, adx_pair, ffm_block


class EmaCrossStrategy(Strategy):
    name = "ema"
    model_filename = "ema_cross_chronos.joblib"

    def _fired(self, bars):
        c = bars["close"].to_numpy(float)
        ef = ind.ema(c, config.EMA_FAST)
        es = ind.ema(c, config.EMA_SLOW)
        a = ind.adx(bars, config.ADX_P)
        i = len(c) - 1
        if config.ADX_GATE and (not np.isfinite(a[i]) or a[i] < config.ADX_GATE):
            return None                                  # not a trending regime
        if not (np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])):
            return None
        if ef[i - 1] <= es[i - 1] and ef[i] > es[i]:
            return 1                                     # fast crosses above slow
        if ef[i - 1] >= es[i - 1] and ef[i] < es[i]:
            return -1                                    # fast crosses below slow
        return None

    def _hand_features(self, bars, i, direction):
        c = bars["close"].to_numpy(float)
        ef = ind.ema(c, config.EMA_FAST)
        es = ind.ema(c, config.EMA_SLOW)
        atr_i = float(ind.atr(bars, config.ATR_P)[i])
        a = atr_i if (np.isfinite(atr_i) and atr_i > 0) else np.nan
        d = direction

        def g(x):
            return float(x) if np.isfinite(x) else 0.0

        with np.errstate(invalid="ignore"):
            ema_spread = (ef[i] - es[i]) / a * d
            k = config.SLOW_SLOPE_K
            slow_slope = ((es[i] - es[i - k]) / a * d) if i - k >= 0 else np.nan
            price_vs_slow = (c[i] - es[i]) / a * d
        adx_i, adx_slope = adx_pair(bars, i)
        hand = [g(ema_spread), g(slow_slope), g(price_vs_slow),
                adx_i / 100.0, adx_slope / 100.0]
        ffm = ffm_block(bars, i)
        return np.concatenate([ffm, hand]).astype(np.float32)
