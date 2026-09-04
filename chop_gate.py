#!/usr/bin/env python3
"""chop_gate.py — point-in-time market-quality gate for the autotrader.

Validated 2026-08-23 on the repo's own backtest ledgers (281 trades, 5 symbols,
Apr→Aug 2026), walk-forward expanding window refit every 20 trades:

    traded  (atr_ratio < thr): n=87  WR=66%  avgR=+1.31
    skipped (atr_ratio >= thr): n=134 would-be WR=58% avgR=+0.94
    baseline all:              n=221 WR=61% avgR=+1.08

Feature: ATR(14)/ATR(100) on the live 3-min bars — fast vol vs slow vol.
When short-term volatility expands much faster than baseline (>~1.0-1.3,
refit monthly by attribution), the EMA-cross edge degrades (whipsaw chop).

Point-in-time safe: uses only bars up to and including the signal bar.
"""
import os

import numpy as np
import pandas as pd

# Default threshold; attribution.py may retune via .env AUTOTRADE_CHOP_MAX.
DEFAULT_MAX = float(os.environ.get("AUTOTRADE_CHOP_MAX", "1.0"))
MIN_BARS = 120            # need this many bars for a stable slow ATR


def atr_ratio(bars) -> float:
    """ATR(14)/ATR(100) of the bar frame. NaN when there isn't enough history.

    bars: df with high/low/close columns (the broker frame as-is)."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    if len(tr.dropna()) < MIN_BARS:
        return float("nan")
    fast = tr.rolling(14).mean()
    slow = tr.rolling(100).mean()
    if len(fast.dropna()) == 0 or len(slow.dropna()) == 0:
        return float("nan")
    f, s = float(fast.iloc[-1]), float(slow.iloc[-1])
    if np.isnan(s) or np.isnan(f) or s == 0:
        return float("nan")
    return f / s


def should_block(bars, max_ratio: float | None = None) -> tuple[bool, str]:
    """True (+reason) when the current bar context is too choppy to enter.
    Never raises: any internal problem returns False (fail-open) so the gate
    can never take the bot offline by accident."""
    try:
        r = atr_ratio(bars)
        if np.isnan(r):
            return False, ""          # not enough history yet — don't block
        mx = DEFAULT_MAX if max_ratio is None else max_ratio
        if r >= mx:
            return True, f"chop gate ATR14/ATR100={r:.2f} >= {mx:.2f} (vol whipsaw)"
        return False, ""
    except Exception:
        return False, ""
