#!/usr/bin/env python3
"""strategies/pattern_lane.py — proven 30-min candle-pattern lane (bear-side).

The ML strategies grade entries with a trained model; this lane grades by the
HISTORICAL win rate of the 30-min candle pattern that just closed. Measured
2026-08-18 across all 5 symbols (Apr-Jun 2026 window, 2h forward labels):

    MARUBOZU BEAR      55.0%   <- strongest
    BEAR ENGULFING     53.2%
    SHOOTING STAR      52.6%
    HIGH-VOL BEAR WICK 52.1%
    (bull-side patterns 48.8-50.4% — NO edge, so this lane is SHORT-only.)

Freshness: fires only within the first FRESH_MIN minutes of a new 30-min
bucket — a stale pattern is somebody else's trade. Every entry still goes
through the LLM veto (fail-closed) and the fixed 2.0R bracket.
"""
import numpy as np
import pandas as pd

import config
import indicators as ind
from strategies.base import Strategy
from candle_patterns import resample_30m, detect_on_candle

# proba sentinel: 0.40 sits inside [PROBA_FLOOR, evolve ceil-floor] so the
# adaptive confidence ceiling can never starve the lane; the LLM veto is the
# real judge of each entry.
PATTERN_PROBA = {
    "MARUBOZU BEAR": 0.40,
    "BEAR ENGULFING": 0.40,
    "SHOOTING STAR": 0.40,
    "HIGH-VOL BEAR WICK": 0.40,
}
# explicit priority when a candle prints several patterns at once
PATTERN_PRIORITY = {"MARUBOZU BEAR": 4, "BEAR ENGULFING": 3,
                    "SHOOTING STAR": 2, "HIGH-VOL BEAR WICK": 1}
# fire ONLY on the bar that closes the 30m candle (minute % 30 == 0):
# one entry per pattern, entered at the pattern candle's close (no repaint,
# no double/triple exposure from re-firing the same candle on later bars)
FIRE_ON_CLOSE = True
R_HAT = 0.6              # conservative historical avg R per pattern trade
TAIL = 1200              # bars window for resample (60h -> 120 candles)


class PatternLaneStrategy(Strategy):
    name = "pattern"
    model_filename = ""            # no ML bundle — graded by pattern stats + veto

    def _fired(self, bars):
        # Symbol gate (config.PATTERN_SYMBOLS): the lane's edge was only ever
        # proven on NQ (see config.py notes). Other symbols' pattern shorts
        # bled the eval live 2026-08-19 (3 of 4 non-NQ signals lost -1R).
        allowed = getattr(config, "PATTERN_SYMBOLS", None)
        if allowed is not None and config.SYMBOL not in allowed:
            return None
        if len(bars) < 120:
            return None
        d = resample_30m(bars.tail(TAIL).reset_index(drop=True))
        if len(d) < 4:
            return None
        last_t = bars["time"].iloc[-1]
        if not hasattr(last_t, "minute"):
            return None
        if (last_t.minute % 30) != 0:
            return None                        # only the 30m-close bar fires
        # d.iloc[-1] = in-progress 30m bucket; d.iloc[-2] = just-closed candle
        cur, prv = d.iloc[-2], d.iloc[-3]
        vol_ma = d["volume"].rolling(20, min_periods=5).mean()
        v = float(cur["volume"]) if pd.notna(cur["volume"]) else None
        vm = float(vol_ma.iloc[-2]) if pd.notna(vol_ma.iloc[-2]) else None
        pats = detect_on_candle(cur["open"], cur["high"], cur["low"],
                                cur["close"], prv["open"], prv["close"],
                                prv["high"], prv["low"], vol=v, vol_ma=vm)
        winners = [p for p in pats if p in PATTERN_PROBA]
        if not winners:
            return None
        # strongest pattern wins (MARUBOZU BEAR > BEAR ENGULFING > ...)
        self._last_pattern = max(winners, key=PATTERN_PRIORITY.get)
        return -1                              # bear-side lane

    def detect(self, bars):
        sig = super().detect(bars)
        if sig is not None:
            sig.pattern = getattr(self, "_last_pattern", "?")
            # 2026-08-19 stop-sizing finding: the inherited base stop
            # (0.5×ATR(20) of the 3-min bars) was suspected of being too tight
            # for 30m pattern candles. Grid backtest (NQ, Apr-Jun 2026) showed
            # WIDENING DOES NOT HELP (0.75x/1.0x were neutral-to-worse) — the
            # live bleed came from non-NQ symbols + the GC tick bug + sample
            # noise, not stop width. The multiplier stays configurable but
            # defaults to 0.5 (= the base stop; this override is then a no-op,
            # with a floor so it can never go tighter than the ML lanes).
            a = float(ind.atr(bars, config.ATR_P)[sig.bar_index])
            if np.isfinite(a) and a > 0:
                risk = max(config.PATTERN_STOP_MULT * a, config.STOP_ATR * a)
                sig.risk = risk
                sig.stop = sig.entry - sig.direction * risk
        return sig

    def grade(self, bars, sig, emb=None):
        return PATTERN_PROBA.get(getattr(sig, "pattern", "?"), 0.40), R_HAT

    def _hand_features(self, bars, i, direction):
        return np.zeros(1, dtype=np.float32)   # unused — grade() is overridden

    def has_model(self):
        return True                            # pattern stats + veto, no bundle
