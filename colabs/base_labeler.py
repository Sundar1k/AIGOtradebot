#!/usr/bin/env python3
"""colabs/base_labeler.py — shared machinery for the reconstructed labelers.

Implements the StrategyLabeler protocol on top of the CURRENT strategy code:

  calendar()      — long frame [item_id, timestamp, target] over all tickers
  build(lo,hi,ts) — causal signals in [lo,hi): (contexts, labels, keys)
  features(keys)  — 76 FFM + per-strategy handcrafts (direction-signed)
  evaluate(keys, preds) — settle each taken trade from bar extremes -> R
                       (preds: 1 = take the trade, 0 = skip; cost included)

Label semantics (matches shipped bundles, PROBA_MEANING='P(trade reaches TP
before SL)'): binary 2-class — label 1 if the trade hits the +2R target
before the 0.5×ATR stop, else 0. Outcomes resolve forward bar-by-bar from
bar extremes (stop assumed first on same-bar touch, same as sim_broker).
Decisions whose outcome window reaches >= test_start are purged (no
train->val/test leakage).

Contexts are log-close windows of length ctx_window (128) ending at the
signal bar — the exact input backbone.embed expects (embed_worker builds
the same windows from the same closes).
"""
import json
import os

import numpy as np
import pandas as pd

import config
import indicators as ind
from strategies.base import ffm_block

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SYMBOLS = [("NQ", "NQ_3min.csv", 10.0), ("ES", "ES_3min.csv", 4.0),
           ("GC", "GC_3min.csv", 2.5), ("RTY", "RTY_3min.csv", 8.0),
           ("YM", "YM_3min.csv", 6.0)]

CTX = getattr(config, "CTX", 128)              # context window (bars)
STOP_ATR = getattr(config, "STOP_ATR", 0.5)    # stop distance in ATR
RR = 2.0                                       # 2R target (labeler convention)


def _load_bars(ticker: str, tf: str) -> pd.DataFrame:
    """Load one ticker's bars, UTC DatetimeIndex, columns o/h/l/c/v."""
    fn = f"{ticker}_{tf}.csv"
    df = pd.read_csv(os.path.join(DATA_DIR, fn))
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df["time"] = df.index
    return df


class BaseLabeler:
    """Shared protocol glue; subclasses define _fired(i) and _hand(i, d)."""

    n_classes = 2
    tf = "3min"                      # override in subclass for 15min

    def __init__(self):
        self._b = {}                 # (ticker, tf) -> bars df
        self._sig_cache = {}         # (ticker, tf) -> {i: direction} all signals
        for tk, fn, _ in SYMBOLS:
            self._b[(tk, self.tf)] = _load_bars(tk, self.tf)
        for key, df in self._b.items():
            self._sig_cache[key] = self._scan(key, df)

    # ── subclass hooks ──────────────────────────────────────────────
    def _fired(self, key, df, i):    # -> +1/-1/None at bar i
        raise NotImplementedError

    def _hand(self, key, df, i, d):  # -> np.ndarray handcrafts at bar i
        raise NotImplementedError

    def handcraft_names(self):
        raise NotImplementedError

    # ── scanning (no look-ahead; uses bars <= i) ────────────────────
    def _scan(self, key, df):
        n = len(df)
        sigs = {}
        # per-ticker precomputed arrays (cache on the df)
        cache = getattr(df, "_ff_cache", None)
        if cache is None:
            c = df["close"].to_numpy(float)
            hi = df["high"].to_numpy(float)
            lo = df["low"].to_numpy(float)
            cache = {"c": c, "hi": hi, "lo": lo,
                     "atr": np.asarray(ind.atr(df, config.ATR_P), dtype=float),
                     "ef": ind.ema(c, config.EMA_FAST),
                     "es": ind.ema(c, config.EMA_SLOW),
                     "adx": np.asarray(ind.adx(df, config.ADX_P), dtype=float)}
            df._ff_cache = cache
        warm = max(CTX + 2, config.ADX_P * 3, 2 * getattr(config, "GANN_SWING_K", 5) + 1)
        for i in range(warm, n):
            d = self._fired(key, df, i)
            if d is not None:
                sigs[i] = d
        return sigs

    # ── protocol: calendar ──────────────────────────────────────────
    def calendar(self) -> pd.DataFrame:
        rows = []
        for (tk, tf), df in self._b.items():
            for i in self._sig_cache[(tk, tf)]:
                rows.append({"item_id": tk, "timestamp": df.index[i],
                             "target": 0})
        return pd.DataFrame(rows)

    # ── protocol: build ─────────────────────────────────────────────
    def build(self, lo, hi, test_start):
        """(contexts, labels, keys) for signals with timestamp in [lo, hi)."""
        lo = pd.Timestamp(lo)
        hi = pd.Timestamp(hi)
        contexts, labels, keys = [], [], []
        for (tk, tf), df in self._b.items():
            sigs = self._sig_cache[(tk, tf)]
            c = df["close"].to_numpy(float)
            idx = df.index
            for i, d in sigs.items():
                t = idx[i]
                if t < lo or t >= hi:
                    continue
                # outcome window: resolve forward from bar i+1
                if test_start is not None and t >= pd.Timestamp(test_start):
                    continue
                # context: log-close window ending at i (CTX bars)
                if i - CTX + 1 < 0:
                    continue
                w = np.log(np.clip(c[i - CTX + 1:i + 1], 1e-6, None))
                # label: 2R target before 0.5xATR stop, resolved bar-by-bar
                lab, mfe_r = self._outcome(df, i, d)
                if lab is None:           # unresolved before data end / purge
                    continue
                contexts.append(w.astype(np.float32))
                labels.append(lab)
                # key = (ticker, bar_idx, 0, mfe_r) — k[3] feeds the risk head
                # (produce.py Stage 5: max_rr from k[3]), matching the shipped
                # bundles' risk_head (peak-R̂ regression).
                keys.append((tk, i, 0, mfe_r))
        return contexts, np.asarray(labels, np.int64), keys

    def _outcome(self, df, i, d):
        """(label, mfe_r): +1 if 2R target hit before 0.5×ATR stop; 0 if stop
        first; (None, None) if neither resolved (open at data end). Stop
        assumed first on a same-bar touch (conservative, sim_broker
        convention). mfe_r = max favorable excursion in R before resolution
        (feeds the risk head, same as the shipped bundles)."""
        c = df["close"].to_numpy(float)
        hi = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        atr = getattr(df, "_ff_cache")["atr"]
        n = len(df)
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            return None, None
        entry = float(c[i])
        risk = STOP_ATR * a
        stop = entry - d * risk
        tgt = entry + d * RR * risk
        mfe = 0.0
        for j in range(i + 1, n):
            if d > 0:
                mfe = max(mfe, (hi[j] - entry) / risk)
                if lo[j] <= stop:
                    return 0, float(mfe)
                if hi[j] >= tgt:
                    return 1, float(mfe)
            else:
                mfe = max(mfe, (entry - lo[j]) / risk)
                if hi[j] >= stop:
                    return 0, float(mfe)
                if lo[j] <= tgt:
                    return 1, float(mfe)
        return None, None

    # ── protocol: features ──────────────────────────────────────────
    def features(self, keys):
        """(N, 76 + n_hand) direction-signed feature rows (NaN kept for
        XGBoost). Matches the shipped bundle width via feat_dim check."""
        rows = []
        for key in keys:
            tk, i = key[0], key[1]
            df = self._b[(tk, self.tf)]
            d = self._sig_cache[(tk, self.tf)][i]
            rows.append(self._feat_row(df, i, d))
        return np.asarray(rows, np.float32)

    def _feat_row(self, df, i, d):
        ffm = self._ffm_cache(df)
        key = next(((tk, tf) for (tk, tf), d2 in self._b.items() if d2 is df), None)
        hand = self._hand(key, df, i, d)
        return np.concatenate([ffm[i], hand]).astype(np.float32)

    def _ffm_cache(self, df):
        """Full 76-col FFM matrix for this df, computed ONCE (derive_features
        is whole-frame; per-row calls are ~1000x slower). Filtered to
        FFM_COLS in order — the derived frame carries extra numeric cols."""
        if not hasattr(df, "_ffm_full"):
            from strategies.base import FFM_COLS
            from futures_foundation.features import derive_features
            df2 = df.reset_index(drop=True).rename(columns={"time": "datetime"})
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats = derive_features(
                    df2, instrument=config.base_symbol(config.SYMBOL),
                    atr_period=config.ATR_P)
            keep = [c for c in FFM_COLS if c in feats.columns]
            df._ffm_full = feats[keep].to_numpy(np.float32)
        return df._ffm_full

    def feature_names(self):
        # FFM lib columns are not stored on disk; approximate with the
        # ordered names the strategies use (only used for regime/changepoint
        # labeling, not for training).
        cols = [f"ffm_{k}" for k in range(76)] + list(self.handcraft_names())
        return cols

    # ── protocol: evaluate ──────────────────────────────────────────
    def evaluate(self, keys, preds, risk_preds=None):
        """Realized R per key. preds==1 -> take the trade, else skip (R=0).
        Cost: 0.25pt slippage per side (same convention as the live bot's
        risk model), converted to R via each trade's risk. `risk_preds`
        (peak-R̂ estimate per key) is accepted for API compatibility with
        produce.py's dynamic-TP sweep; the fixed-TP convention here is the
        one that ships (risk_head predicts R for the bot's r_hat only)."""
        R = np.zeros(len(keys), float)
        for k, key in enumerate(keys):
            if preds[k] != 1:
                continue
            tk, i = key[0], key[1]
            df = self._b[(tk, self.tf)]
            d = self._sig_cache[(tk, self.tf)][i]
            r = self._settle_r(df, i, d)
            if r is not None:
                R[k] = r
        return R

    def _settle_r(self, df, i, d):
        c = df["close"].to_numpy(float)
        hi = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        atr = getattr(df, "_ff_cache")["atr"]
        n = len(df)
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            return None
        entry = float(c[i])
        risk = STOP_ATR * a
        stop = entry - d * risk
        tgt = entry + d * RR * risk
        for j in range(i + 1, n):
            if d > 0 and lo[j] <= stop:
                return (stop - entry) / risk
            if d < 0 and hi[j] >= stop:
                return (entry - stop) / risk
            if d > 0 and hi[j] >= tgt:
                return (tgt - entry) / risk
            if d < 0 and lo[j] <= tgt:
                return (entry - tgt) / risk
        return None


# make _feat_row work: give each df its ticker for _hand()
for _tk, _fn, _ in SYMBOLS:
    pass


def attach_ticker(df, tk):
    df.name_tk = tk
    return df
