#!/usr/bin/env python3
"""strategies/gann_lane.py — Gann 1x1 angle-line strategy (mechanical).

Gann's 45° (1x1) angle: price above the ascending line from the last
swing low is bullish, below the descending line from the last swing high
is bearish. Normalized per-symbol via ATR (1x1 = GANN_SLOPE_MULT x ATR
per bar). Entry on the direction FLIP of the active angle line.

State-free per bar: recomputes the anchor swing + line from bars (same
pattern as supertrend._fired), so it plugs into the existing Strategy
interface and the backtest harness unchanged.

Hand features mirror supertrend (adx + ffm) so a Chronos grader can be
trained later if the raw lane survives the Phase-4 gate.
"""
import os

import numpy as np

import config
import indicators as ind
from strategies.base import Strategy, adx_pair, ffm_block

SWING_K = getattr(config, "GANN_SWING_K", 5)          # fractal window
SLOPE_MULT = getattr(config, "GANN_SLOPE_MULT", 0.5)  # 1x1 = 0.5 ATR/bar


class GannAngleStrategy(Strategy):
    name = "gann"
    # ML bundle (2026-08-22): the 15-min Phase-C train produced
    # gann_chronos_15min.joblib — the lane now uses the base Strategy
    # grade()/model_path() machinery like ema/st/orb (proba + r_hat from
    # XGBoost) instead of the hardcoded 0.40 default. 3-min has no bundle
    # yet, so on 3-min the lane falls back to the default proba path via
    # _load_bundle → has_model() False → default grade (see base.Strategy).
    model_filename = "gann_chronos.joblib"

    def _swings(self, bars):
        """Last confirmed swing low/high (fractal of ±SWING_K bars)."""
        lo = bars["low"].to_numpy(float)
        hi = bars["high"].to_numpy(float)
        n = len(lo)
        low_i, high_i = -1, -1
        for i in range(n - SWING_K - 1, SWING_K - 1, -1):
            if low_i < 0 and lo[i] == lo[i - SWING_K:i + SWING_K + 1].min():
                low_i = i
            if high_i < 0 and hi[i] == hi[i - SWING_K:i + SWING_K + 1].max():
                high_i = i
            if low_i >= 0 and high_i >= 0:
                break
        return low_i, high_i

    def _fired(self, bars):
        c = bars["close"].to_numpy(float)
        lo = bars["low"].to_numpy(float)
        hi = bars["high"].to_numpy(float)
        atr = np.asarray(ind.atr(bars, config.ATR_P), dtype=float)
        i = len(c) - 1
        if i < 2 * SWING_K + 1 or not np.isfinite(atr[i]) or atr[i] <= 0:
            return None
        low_i, high_i = self._swings(bars)
        if low_i < 0 or high_i < 0:
            return None
        # 1x1 angle lines from the anchors (ATR-normalized slope)
        line_up = lo[low_i] + SLOPE_MULT * atr[i] * (i - low_i)
        line_dn = hi[high_i] - SLOPE_MULT * atr[i] * (i - high_i)
        # direction state from the previous bar
        prev_up = c[i - 1] > (lo[low_i] + SLOPE_MULT * atr[i - 1] * (i - 1 - low_i))
        prev_dn = c[i - 1] < (hi[high_i] - SLOPE_MULT * atr[i - 1] * (i - 1 - high_i))
        cur_up = c[i] > line_up
        cur_dn = c[i] < line_dn
        if prev_up and not cur_up and not cur_dn:
            return -1                    # lost the ascending angle
        if prev_dn and not cur_dn and not cur_up:
            return 1                     # lost the descending angle
        if not prev_up and not prev_dn:
            # flat → first decisive break of either angle
            if cur_up:
                return 1
            if cur_dn:
                return -1
        return None

    def grade(self, bars, sig, emb=None):
        # ML-graded when a bundle exists for the active timeframe; else
        # FAIL-CLOSED. The legacy constant (0.40, 0.0) landed INSIDE the
        # [0.35, 0.50] confidence band, so every bundle-less gann flip passed
        # the proba gate and traded with no model grading at all — the 2 live
        # -1R losses on 2026-08-22 were exactly this. Raising means a
        # bundle-less gann can never grade (the signal is skipped fail-closed;
        # make_strategies() also refuses to instantiate a bundle-less active
        # strategy at startup). Re-enable only once gann_chronos.joblib is
        # trained for the active timeframe.
        if self.has_model():
            return super().grade(bars, sig, emb)
        raise RuntimeError(
            "gann has no %d-min model (bundle missing) — refusing to grade "
            "a signal with a constant proba" % config.TIMEFRAME_MIN)

    def has_model(self):
        # Only True when the timeframe-matched bundle actually exists
        # (model_filename set 2026-08-22 → gann_chronos_<tf>min.joblib).
        try:
            return os.path.exists(self.model_path())
        except Exception:
            return False

    def _hand_features(self, bars, i, direction):
        adx_i, adx_slope = adx_pair(bars, i)
        ffm = ffm_block(bars, i)
        return np.concatenate([ffm, [adx_i, adx_slope]]).astype(np.float32)
