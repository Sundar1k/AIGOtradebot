#!/usr/bin/env python3
"""strategies/supertrend.py — SuperTrend flip strategy.

A SuperTrend direction flip (ATR-band crossover) is the mechanical entry;
the SuperTrendChronos model grades whether the flip rides or chops.
Hand features = 76 FFM + [adx, adx_slope]  (78 total → feat_dim 334).
"""
import numpy as np

import config
import indicators as ind
from strategies.base import Strategy, adx_pair, ffm_block


class SuperTrendStrategy(Strategy):
    name = "supertrend"
    model_filename = "supertrend_chronos.joblib"

    def _fired(self, bars):
        _line, direction = ind.supertrend(bars, config.ST_PERIOD, config.ST_MULT)
        if direction[-1] == direction[-2]:
            return None                              # no flip on the last bar
        return int(direction[-1])

    def _hand_features(self, bars, i, direction):
        adx_i, adx_slope = adx_pair(bars, i)
        ffm = ffm_block(bars, i)
        return np.concatenate([ffm, [adx_i, adx_slope]]).astype(np.float32)
