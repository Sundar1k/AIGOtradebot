#!/usr/bin/env python3
"""strategies/cisd_ote_detect.py — CISD+OTE zone detection + signal extraction.

VERBATIM port of the training pipeline (Futures-Foundation-Model
colabs/cisd_ote_optuna.py + cisd_ote_chronos.py) so the live signals/features are
byte-identical to how `cisd_ote_chronos.joblib` was trained — no drift. The CISD
displacement + OTE-fib logic is implemented inline exactly as the colab does it
(it does NOT use futures_foundation's newer detect_cisd_signals/compute_ote_zones,
so neither do we). Only mechanical adaptations: the Optuna params are frozen here
(from configs/best_cisd_ote_params.json — training used these fixed values, never
Optuna at train time), and the entry-loop's `_oc.MIN_RISK_FRAC` is the constant
below. Do not "improve" this file — it must mirror training.
"""
from collections import deque

import numpy as np
import pandas as pd

# ── frozen strategy params (best_cisd_ote_params.json) ──────────────────────
PARAMS = {
    "cisd_tf": 12, "swing_period": 3, "tolerance": 0.5, "expiry_bars": 9,
    "liquidity_lookback": 5, "fib_1": 0.5, "fib_2": 0.705,
    "disp_body_ratio_min": 0.3, "disp_close_str_min": 0.4, "rr_target": 5.0,
    "require_sweep": False, "session": "ny", "entry_mode": "bot", "sl_mode": "pivot",
}
ENTRY_TF_MIN = 3                 # entry resolution is always the raw 3-min bar
BASE_MIN = 3                     # default 3-min base timeframe
CISD_TF = int(round(PARAMS["cisd_tf"] * BASE_MIN / 3.0))      # 12 @ 3-min
_htf_min = round(5 * CISD_TF)                                 # HTF = 5× zone
HTF_TF = f"{_htf_min // 60}h" if _htf_min % 60 == 0 else f"{_htf_min}min"   # 1h @ 3-min
N_GEOM = 11                      # 8 OTE-geometry + 2 HTF-trend + 1 session-id
MIN_RISK_FRAC = 1e-4
ATR_P = 20
CTX = 128


# ── resample / pivots (verbatim from cisd_ote_optuna.py) ────────────────────

def resample(df3, minutes):
    """Resample 3-min OHLCV to `minutes`-min bars for zone detection."""
    if minutes == ENTRY_TF_MIN:
        return df3
    out = df3.resample(f"{minutes}min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"])
    return out


def detect_pivots(highs, lows, period):
    """Pivot highs/lows: a bar is the unique extreme of its (2*period+1) window."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(highs)
    if n < 2 * period + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    win_h = sliding_window_view(highs, 2 * period + 1)
    win_l = sliding_window_view(lows, 2 * period + 1)
    ch, cl = win_h[:, period], win_l[:, period]
    is_ph = (ch == win_h.max(1)) & (np.sum(win_h == ch[:, None], 1) == 1)
    is_pl = (cl == win_l.min(1)) & (np.sum(win_l == cl[:, None], 1) == 1)
    return np.where(is_ph)[0] + period, np.where(is_pl)[0] + period


# ── zone detection (verbatim from cisd_ote_optuna.detect_zone_signals) ──────

def detect_zone_signals(df_zone, p):
    """CISD displacement detection on zone-timeframe bars. Returns the zone frame
    with: cisd_signal (0/1 bear /2 bull), fib_top, fib_bot, origin, had_sweep,
    disp_strength."""
    o = df_zone["open"].values.astype(np.float64)
    h = df_zone["high"].values.astype(np.float64)
    l = df_zone["low"].values.astype(np.float64)
    c = df_zone["close"].values.astype(np.float64)
    n = len(c)

    sw = p["swing_period"]
    tol = p["tolerance"]
    exp = p["expiry_bars"]
    liq = p["liquidity_lookback"]
    f1, f2 = p["fib_1"], p["fib_2"]
    brmin, csmin = p["disp_body_ratio_min"], p["disp_close_str_min"]

    piv_high_bars, piv_low_bars = detect_pivots(h, l, sw)
    sh_by_conf, sl_by_conf = {}, {}
    for b in piv_high_bars:
        cf = b + sw
        if cf < n:
            sh_by_conf.setdefault(cf, []).append((h[b], b))
    for b in piv_low_bars:
        cf = b + sw
        if cf < n:
            sl_by_conf.setdefault(cf, []).append((l[b], b))

    cisd_signal = np.zeros(n, dtype=np.int8)
    fib_top = np.full(n, np.nan)
    fib_bot = np.full(n, np.nan)
    origin = np.full(n, np.nan)         # swing origin of the leg (SL anchor)
    had_sweep = np.zeros(n, dtype=np.int8)
    disp_strength = np.zeros(n, dtype=np.float32)

    active_sh, active_sl = deque(), deque()
    last_wicked_high = last_wicked_low = -10 ** 9
    bear_pots, bull_pots = deque(), deque()

    for bar in range(1, n):
        if bar in sh_by_conf:
            for pr, b in sh_by_conf[bar]:
                active_sh.append((pr, b))
        if bar in sl_by_conf:
            for pr, b in sl_by_conf[bar]:
                active_sl.append((pr, b))

        new_sh = deque()
        for pr, b in active_sh:
            if bar - b >= exp:
                continue
            if h[bar] >= pr:
                last_wicked_high = bar
            else:
                new_sh.append((pr, b))
        active_sh = new_sh

        new_sl = deque()
        for pr, b in active_sl:
            if bar - b >= exp:
                continue
            if l[bar] <= pr:
                last_wicked_low = bar
            else:
                new_sl.append((pr, b))
        active_sl = new_sl

        if c[bar - 1] < o[bar - 1] and c[bar] > o[bar]:
            bear_pots.append((o[bar], bar))
        if c[bar - 1] > o[bar - 1] and c[bar] < o[bar]:
            bull_pots.append((o[bar], bar))
        while bear_pots and bar - bear_pots[0][1] >= exp:
            bear_pots.popleft()
        while bull_pots and bar - bull_pots[0][1] >= exp:
            bull_pots.popleft()

        # ── Bearish CISD ──
        while bear_pots:
            pot_price, pot_bar = bear_pots[0]
            if c[bar] < pot_price:
                highest_c = c[pot_bar:bar + 1].max()
                top_level = 0.0
                idx = pot_bar + 1
                while idx < bar and c[idx] < o[idx]:
                    top_level = o[idx]; idx += 1
                if top_level > 0 and (top_level - pot_price) > 0:
                    ratio = (highest_c - pot_price) / (top_level - pot_price)
                    if ratio > tol:
                        fr = h[bar] - l[bar]
                        body = abs(c[bar] - o[bar])
                        br = body / fr if fr > 0 else 0.0
                        cs = (h[bar] - c[bar]) / fr if fr > 0 else 0.0
                        if br >= brmin and cs >= csmin:
                            cisd_signal[bar] = 1
                            disp_strength[bar] = ratio
                            if (bar - last_wicked_high) <= liq:
                                had_sweep[bar] = 1
                            h_max = h[pot_bar:bar + 1].max()
                            diff = h_max - l[bar]
                            fib_top[bar] = max(h_max - diff * f1, h_max - diff * f2)
                            fib_bot[bar] = min(h_max - diff * f1, h_max - diff * f2)
                            origin[bar] = h_max          # bear SL anchor = swing high
                            bear_pots.clear(); break
                        bear_pots.popleft(); continue
                    bear_pots.popleft(); continue
                bear_pots.popleft(); continue
            break

        # ── Bullish CISD ──
        while bull_pots:
            pot_price, pot_bar = bull_pots[0]
            if c[bar] > pot_price:
                lowest_c = c[pot_bar:bar + 1].min()
                bottom_level = 0.0
                idx = pot_bar + 1
                while idx < bar and c[idx] > o[idx]:
                    bottom_level = o[idx]; idx += 1
                if bottom_level > 0 and (pot_price - bottom_level) > 0:
                    ratio = (pot_price - lowest_c) / (pot_price - bottom_level)
                    if ratio > tol:
                        fr = h[bar] - l[bar]
                        body = abs(c[bar] - o[bar])
                        br = body / fr if fr > 0 else 0.0
                        cs = (c[bar] - l[bar]) / fr if fr > 0 else 0.0
                        if br >= brmin and cs >= csmin:
                            cisd_signal[bar] = 2
                            disp_strength[bar] = ratio
                            if (bar - last_wicked_low) <= liq:
                                had_sweep[bar] = 1
                            l_min = l[pot_bar:bar + 1].min()
                            diff = h[bar] - l_min
                            fib_top[bar] = max(l_min + diff * f1, l_min + diff * f2)
                            fib_bot[bar] = min(l_min + diff * f1, l_min + diff * f2)
                            origin[bar] = l_min          # bull SL anchor = swing low
                            bull_pots.clear(); break
                        bull_pots.popleft(); continue
                    bull_pots.popleft(); continue
                bull_pots.popleft(); continue
            break

    out = df_zone.copy()
    out["cisd_signal"] = cisd_signal
    out["fib_top"] = fib_top
    out["fib_bot"] = fib_bot
    out["origin"] = origin
    out["had_sweep"] = had_sweep
    out["disp_strength"] = disp_strength
    return out


# ── signal extraction (verbatim from cisd_ote_chronos._extract_signals) ─────

def extract_signals(df3, df_sig, session_mask, atr3=None, max_risk_atr=None):
    """Port of the cisd_ote_optuna entry loop (labeler mode — EVERY fired
    signal). Yields per-signal records: dict(sig_bar, exec_idx, is_long, entry,
    sl, fib_top, fib_bot, zone_age, had_sweep, disp)."""
    o3 = df3["open"].values.astype(np.float64)
    h3 = df3["high"].values.astype(np.float64)
    l3 = df3["low"].values.astype(np.float64)
    c3 = df3["close"].values.astype(np.float64)
    t3 = df3.index.astype(np.int64).values
    n3 = len(c3)

    h15 = df_sig["high"].values.astype(np.float64)
    l15 = df_sig["low"].values.astype(np.float64)
    c15 = df_sig["close"].values.astype(np.float64)
    cisd15 = df_sig["cisd_signal"].values
    ft15 = df_sig["fib_top"].values
    fb15 = df_sig["fib_bot"].values
    or15 = df_sig["origin"].values
    sweep15 = df_sig["had_sweep"].values
    disp15 = df_sig["disp_strength"].values
    t15 = df_sig.index.astype(np.int64).values
    n15 = len(c15)

    sw = PARAMS["swing_period"]
    brmin, csmin = PARAMS["disp_body_ratio_min"], PARAMS["disp_close_str_min"]
    require_sweep = PARAMS["require_sweep"]
    entry_mode, sl_mode = PARAMS["entry_mode"], PARAMS["sl_mode"]
    zone_dt_ns = int(CISD_TF) * 60 * 1_000_000_000

    recs = []
    active = []
    for i15 in range(sw * 3, n15):
        sig = cisd15[i15]
        if sig in (1, 2) and not np.isnan(ft15[i15]):
            active.insert(0, {
                "fib_top": ft15[i15], "fib_bot": fb15[i15], "origin": or15[i15],
                "created": i15, "is_bull": sig == 2, "fired": False,
                "entered": False, "had_sweep": bool(sweep15[i15]),
                "disp": float(disp15[i15])})
            if len(active) > 20:
                active.pop()

        prev_t15 = t15[i15 - 1] if i15 > 0 else 0
        j_start = int(np.searchsorted(t3, prev_t15, side="right"))
        j_end = int(np.searchsorted(t3, t15[i15], side="right"))

        rm = []
        for zi, z in enumerate(active):
            invalid = ((z["is_bull"] and c15[i15] < z["fib_bot"]) or
                       (not z["is_bull"] and c15[i15] > z["fib_top"]))
            if l15[i15] <= z["fib_top"] and h15[i15] >= z["fib_bot"]:
                z["entered"] = True
            if z["entered"] and not z["fired"] and not invalid and i15 > z["created"]:
                if require_sweep and not z["had_sweep"]:
                    if invalid:
                        rm.append(zi)
                    continue
                is_long = z["is_bull"]
                ce = 0.5 * (z["fib_top"] + z["fib_bot"])
                limit = {"top": z["fib_top"], "ce": ce,
                         "bot": z["fib_bot"]}.get(entry_mode)
                if sl_mode == "pivot" and not np.isnan(z["origin"]):
                    sl = z["origin"]
                else:
                    sl = z["fib_bot"] if is_long else z["fib_top"]

                for j3 in range(j_start, min(j_end, n3)):
                    if t3[j3] < t15[z["created"]] + zone_dt_ns:
                        continue                  # zone not yet confirmed (causal)
                    if not session_mask[j3]:
                        continue
                    if not (l3[j3] <= z["fib_top"] and h3[j3] >= z["fib_bot"]):
                        continue
                    if entry_mode == "bounce":
                        fr = h3[j3] - l3[j3]
                        br = abs(c3[j3] - o3[j3]) / fr if fr > 0 else 0.0
                        if is_long:
                            cs = (c3[j3] - l3[j3]) / fr if fr > 0 else 0.0
                            ok = c3[j3] > o3[j3] and br >= brmin and cs >= csmin
                        else:
                            cs = (h3[j3] - c3[j3]) / fr if fr > 0 else 0.0
                            ok = c3[j3] < o3[j3] and br >= brmin and cs >= csmin
                        if not (ok and j3 + 1 < n3):
                            continue
                        entry, exec_idx = o3[j3 + 1], j3 + 1
                    else:
                        if is_long:
                            if l3[j3] > limit:
                                continue
                        else:
                            if h3[j3] < limit:
                                continue
                        if j3 + 1 >= n3:
                            continue
                        entry, exec_idx = o3[j3 + 1], j3 + 1
                    risk = (entry - sl) if is_long else (sl - entry)
                    if risk < MIN_RISK_FRAC * entry:
                        continue
                    if (max_risk_atr and atr3 is not None
                            and atr3[exec_idx] > 0
                            and risk > max_risk_atr * atr3[exec_idx]):
                        continue                       # tight-risk filter
                    z["fired"] = True
                    recs.append(dict(
                        sig_bar=j3, exec_idx=exec_idx, is_long=is_long,
                        entry=float(entry), sl=float(sl),
                        fib_top=float(z["fib_top"]), fib_bot=float(z["fib_bot"]),
                        zone_age=float(i15 - z["created"]),
                        zone_created=int(z["created"]),
                        had_sweep=z["had_sweep"], disp=z["disp"]))
                    break
            if invalid:
                rm.append(zi)
        for zi in reversed(rm):
            if zi < len(active):
                active.pop(zi)
    return recs
