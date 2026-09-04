#!/usr/bin/env python3
"""strategies/base.py — the generic Strategy interface.

A Strategy (a) DETECTS its mechanical entry on the latest closed bar and
(b) GRADES it with its own pre-trained Chronos+XGBoost model (a joblib bundle
from futures_foundation). Concrete strategies live in sibling files and inherit
from `Strategy`; the bot can run one or several at once.

Trade definition matches the models' training: entry ≈ next-bar fill, stop =
STOP_ATR × ATR(ATR_P). The model only learns SELECTION; direction is mechanical.

Inference per signal bar i:  X = concat([embed_256, hand]) → heads
  embed  = futures_foundation.foundation.embed_bars(closes, [i])   (subprocess)
  hand   = 76 FFM features (live, in the models' parquet column order) + the
           strategy's public handcrafts (adx/adx_slope, or the 5 EMA features)
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import config
import indicators as ind

# FFM feature columns in the EXACT order the models were trained on (extracted
# from the training parquet). Live values are placed by name into this order;
# any column the current library doesn't produce stays NaN (XGBoost handles it).
with open(config.FFM_COLUMNS_PATH) as _f:
    FFM_COLS = json.load(_f)


@dataclass
class Signal:
    strategy: str
    direction: int          # +1 long / -1 short
    entry: float            # signal-bar close (≈ live fill)
    stop: float             # protective stop price
    risk: float             # |entry - stop| in price (= STOP_ATR × ATR)
    bar_index: int
    bar_time: object
    proba: float = 0.0
    r_hat: float = 0.0


def embed_context(bars: pd.DataFrame, i: int) -> np.ndarray:
    """Chronos context embedding (1, 256) for the window ending at bar i. All
    strategies firing on the same bar share this same context, so it is computed
    once per bar and reused across strategies. Routed through the warm embedding
    worker (model loaded once per session)."""
    import embedder
    return embedder.embed_bars(bars["close"].to_numpy(float), [i], ctx=config.CTX)


def ffm_block(bars: pd.DataFrame, i: int) -> np.ndarray:
    """76 FFM features at bar i, in the models' parquet column order. Computed
    live via futures_foundation.derive_features; absent columns → NaN.
    `i=None` returns the whole (N, 76) matrix (derive ONCE — labelers use
    this; the per-bar path below is byte-identical to the original)."""
    from futures_foundation.features import derive_features

    if i is None:
        df = bars
    else:
        df = bars.rename(columns={"time": "datetime"})
    feats = derive_features(df, instrument=config.base_symbol(config.SYMBOL),
                            atr_period=config.ATR_P)
    if i is None:
        num = feats.select_dtypes(include=[np.number])
        return num.to_numpy(np.float32)
    row = feats.iloc[i]
    cols = feats.columns
    out = np.full(len(FFM_COLS), np.nan, dtype=np.float32)
    for k, name in enumerate(FFM_COLS):
        if name in cols:
            val = row[name]
            if pd.notna(val):          # leave NaN for absent/NA (XGBoost handles it)
                out[k] = val
    return out


def adx_pair(bars: pd.DataFrame, i: int):
    """(adx, adx_slope) at bar i — the public handcrafts both models share."""
    a = ind.adx(bars, config.ADX_P)
    adx_i = float(a[i]) if np.isfinite(a[i]) else 0.0
    k = config.ADX_SLOPE
    if i >= k and np.isfinite(a[i]) and np.isfinite(a[i - k]):
        slope = float(a[i] - a[i - k])
    else:
        slope = 0.0
    return adx_i, slope


def recent_jump(bars: pd.DataFrame, i: int, lookback: int = 2) -> bool:
    """True if bar i or any of the `lookback` bars before it is a "jump" — a
    |close-to-close| move larger than JUMP_ATR_MULT × ATR(ATR_P). Point-in-time
    (uses only bars <= i). Jump moves are theoretically unpredictable; the
    factor-zoo paper drops them before modeling. Disabled when JUMP_ATR_MULT <= 0.
    """
    if config.JUMP_ATR_MULT <= 0:
        return False
    a = ind.atr(bars, config.ATR_P)
    c = bars["close"].to_numpy(float)
    for j in range(max(1, i - lookback), i + 1):
        av = a[j]
        if not (np.isfinite(av) and av > 0):
            continue
        if abs(c[j] - c[j - 1]) > config.JUMP_ATR_MULT * av:
            return True
    return False


class Strategy(ABC):
    """Generic strategy: detect a mechanical entry, then grade it with a model."""

    name: str = "strategy"
    model_filename: str = ""

    def __init__(self):
        self._bundle = None                 # lazy joblib load

    # ── signal detection (subclass-specific) ───────────────────────────
    @abstractmethod
    def _fired(self, bars: pd.DataFrame) -> Optional[int]:
        """Return the trade direction (+1/-1) if the last closed bar is an
        entry for this strategy, else None."""

    @abstractmethod
    def _hand_features(self, bars: pd.DataFrame, i: int, direction: int) -> np.ndarray:
        """The strategy's hand-crafted feature vector at bar i (FFM + handcrafts)."""

    # ── shared entry construction ──────────────────────────────────────
    def detect(self, bars: pd.DataFrame) -> Optional[Signal]:
        d = self._fired(bars)
        if d is None:
            return None
        i = len(bars) - 1
        a = float(ind.atr(bars, config.ATR_P)[i])
        if not np.isfinite(a) or a <= 0:
            return None
        entry = float(bars["close"].iloc[i])
        risk = config.STOP_ATR * a                  # stop distance in price
        stop = entry - d * risk
        return Signal(self.name, d, entry, stop, risk, i, bars["time"].iloc[i])

    # ── grading (shared) ───────────────────────────────────────────────
    def grade(self, bars: pd.DataFrame, sig: Signal, emb=None):
        """(proba, r_hat) from this strategy's model for the detected signal.

        `emb` is the Chronos context embedding; strategies firing on the same bar
        share the SAME context, so the caller computes it once (embed_context)
        and passes it in — one Chronos pass per bar, not per strategy."""
        if emb is None:
            emb = embed_context(bars, sig.bar_index)
        hand = self._hand_features(bars, sig.bar_index, sig.direction).reshape(1, -1)
        X = np.concatenate([emb, hand], axis=1).astype(np.float32)

        bundle = self._load_bundle()
        proba = float(bundle["signal_head"].predict_proba(X)[0, 1])
        risk_head = bundle.get("risk_head")
        # r_hat is VERIFIED NOISE (corr +0.026 vs realized R, 2026-08-17) —
        # informational only, never gate on it. Kept for log/state-line
        # backward-compat (the veto prompt is byte-pinned to it).
        r_hat = float(risk_head.predict(X)[0]) if risk_head is not None else 0.0
        return proba, r_hat

    def model_path(self) -> str:
        """The model bundle for the active timeframe. The trained default (3-min)
        uses the plain filename; any other timeframe REQUIRES a `_<tf>min` variant
        (e.g. supertrend_chronos_1min.joblib) — no cross-timeframe fallback, so a
        strategy without a matching-timeframe model is simply unavailable."""
        fn = self.model_filename
        if config.TIMEFRAME_MIN != config.TRAINED_TIMEFRAME_MIN:
            base, ext = os.path.splitext(fn)
            return os.path.join(config.MODELS_DIR,
                                f"{base}_{config.TIMEFRAME_MIN}min{ext}")
        return os.path.join(config.MODELS_DIR, fn)

    def has_model(self) -> bool:
        """Whether this strategy has a model for the active timeframe."""
        return os.path.exists(self.model_path())

    def _load_bundle(self) -> dict:
        if self._bundle is None:
            # Importing the pipeline subpackage installs the legacy 'pipelines.chronos'
            # pickle-compat alias so older bundles unpickle without their origin repo.
            # (chronos was renamed to pipeline — fall back to the old name.)
            try:
                import futures_foundation.pipeline  # noqa: F401
            except ModuleNotFoundError:
                import futures_foundation.chronos    # noqa: F401
            self._bundle = joblib.load(self.model_path())
        return self._bundle
