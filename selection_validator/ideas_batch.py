"""ideas_batch.py — cycle 7: four cheap selection ideas (spec 07).

Gap filter (2 variants), jump filter (1.5x ATR20), volume confirmation,
first-hour narrowing — each tested ON TOP of the current live funnel
(window + floor 0.40 + ceil 0.65) with the common pre-registered hold-out
bar: P(ΔavgR>0)>0.95 (bootstrap 10k seed 42), PF_cand>=PF_base, n>=150,
August not worse. Memo-gated; no live changes.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
DATA_GLOB = os.path.join(BASE, "selection_validator", "data", "signals_*.jsonl")
HOLD_START = pd.Timestamp("2025-11-01", tz="UTC")
AUG_START = pd.Timestamp("2026-08-01", tz="UTC")
JUMP_MULT = 1.5


def load_funnel() -> pd.DataFrame:
    from selection_validator.dataset import load_rows
    from selection_validator.dedup_signals import main as dedup
    from selection_validator.time_window import window_mask
    dedup()
    df = load_rows(*glob.glob(DATA_GLOB))
    m = df["take"].astype(bool) & ~df["jump"].astype(bool)
    df = df[m].reset_index(drop=True)
    df["base"] = ((df["proba"] >= 0.40) & (df["proba"] <= 0.65) & window_mask(df))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # per-symbol precomputed arrays (once each, not per row)
    prep: dict[str, dict] = {}

    def _prep(sym):
        if sym not in prep:
            b = pd.read_csv(os.path.join(BASE, "data", f"{sym}_3min.csv")) \
                .rename(columns={"datetime": "time"})
            b["time"] = pd.to_datetime(b["time"], utc=True).reset_index(drop=True)
            et = b["time"].dt.tz_convert("America/New_York")
            prep[sym] = {
                "tarr": b["time"].values.astype("datetime64[ns]"),
                "et_min": (et.dt.hour * 60 + et.dt.minute).to_numpy(),
                "day": et.dt.date.to_numpy(),
                "open": b["open"].to_numpy(), "close": b["close"].to_numpy(),
                "high": b["high"].to_numpy(), "low": b["low"].to_numpy(),
                "volume": b["volume"].to_numpy(),
            }
        return prep[sym]

    gap_align, jump, vol_up, first_hour = ({} for _ in range(4))
    for row in df[["symbol", "ts"]].drop_duplicates().to_dict("records"):
        sym, ts = row["symbol"], row["ts"]
        key = (sym, str(ts))
        try:
            p = _prep(sym)
            j = int(np.searchsorted(p["tarr"], np.datetime64(ts), side="right")) - 1
            if j < 0:
                continue
            # gap: today's open vs yesterday's close
            cur_day = p["day"][j]
            prev = np.where(p["day"] < cur_day)[0]
            if len(prev):
                gap_align[key] = p["open"][j] > p["close"][prev[-1]]
            # jump: max |cc| over last 2 bars vs 1.5 x ATR20 at the signal bar
            lo = max(0, j - 40)
            hi = p["high"][lo:j + 1]; lo_ = p["low"][lo:j + 1]; cl = p["close"][lo:j + 1]
            tr = np.maximum(hi - lo_, np.maximum(np.abs(hi - np.roll(cl, 1)),
                                                 np.abs(lo_ - np.roll(cl, 1))))
            tr[0] = hi[0] - lo_[0]
            atr = pd.Series(tr).rolling(20).mean().iloc[-1]
            cc = abs(p["close"][j] - p["close"][j - 1]) if j >= 1 else 0.0
            if j >= 2:
                cc = max(cc, abs(p["close"][j] - p["close"][j - 2]))
            jump[key] = bool(atr and cc > JUMP_MULT * atr)
            v_now = p["volume"][j]
            v_avg = p["volume"][max(0, j - 20):j].mean()
            vol_up[key] = bool(v_avg and v_now > v_avg)
            m_et = p["et_min"][j]
            first_hour[key] = bool(570 <= m_et < 630)
        except Exception:
            pass
    df["gap_up"] = df.apply(lambda r: gap_align.get((r["symbol"], str(r["ts"])), None), axis=1)
    df["jump_sig"] = df.apply(lambda r: jump.get((r["symbol"], str(r["ts"])), False), axis=1)
    df["vol_up"] = df.apply(lambda r: vol_up.get((r["symbol"], str(r["ts"])), None), axis=1)
    df["first_hour"] = df.apply(lambda r: first_hour.get((r["symbol"], str(r["ts"])), False), axis=1)
    return df


def variants(df: pd.DataFrame) -> dict:
    base = df["base"].astype(bool)
    return {
        "baseline": base,
        "gap_aligned": base & (df["gap_up"] == True),          # noqa: E712
        "gap_opposed": base & (df["gap_up"] == False),         # noqa: E712
        "jump_excluded": base & ~df["jump_sig"],
        "volume_confirmed": base & (df["vol_up"] == True),     # noqa: E712
        "first_hour_only": base & df["first_hour"],
    }


def evaluate(df: pd.DataFrame, mask) -> dict:
    from selection_validator.selectors import stats
    rs = df.loc[mask & df["outcome_r"].notna(), "outcome_r"]
    return stats(rs)


if __name__ == "__main__":
    from selection_validator import harness
    df = load_funnel()
    print(f"funnel: {len(df)} | adding features...", flush=True)
    df = add_features(df)
    hold = df[df["ts"] >= HOLD_START]
    aug = df[df["ts"] >= AUG_START]
    v = variants(df)
    vh = variants(hold)
    va = variants(aug)

    out = {}
    print(f"\n{'variant':18s} {'hold_n':6s} {'hold_avgR':>9s} {'hold_PF':>7s} "
          f"{'P(d>0)':>7s} {'aug':>7s} verdict", flush=True)
    base_hold = evaluate(hold, vh["baseline"])
    base_aug = evaluate(aug, va["baseline"])
    for name, m in v.items():
        if name == "baseline":
            continue
        h = evaluate(hold, vh[name])
        a = evaluate(aug, va[name])
        bm = vh["baseline"] & hold["outcome_r"].notna()
        cm = vh[name] & hold["outcome_r"].notna()
        p = 1 - harness.bootstrap_diff(hold.loc[bm, "outcome_r"],
                                       hold.loc[cm, "outcome_r"])
        aug_ok = (a["avg_r"] or -9) > (base_aug["avg_r"] or -9)
        checks = {"p": p > 0.95, "pf": (h["pf"] or 0) >= (base_hold["pf"] or 0),
                  "n": (h["n"] or 0) >= 150, "aug": aug_ok}
        verdict = "GO" if all(checks.values()) else \
            ("KILL" if (p <= 0.5 or (h["pf"] or 0) < (base_hold["pf"] or 0) or not aug_ok)
             else "INCONCLUSIVE")
        print(f"{name:18s} {h['n']:6d} {h['avg_r'] if h['avg_r'] is not None else 0:9.3f} "
              f"{(h['pf'] or 0):7.2f} {p:7.3f} {a['avg_r'] if a['avg_r'] is not None else 0:7.2f} {verdict}", flush=True)
        out[name] = {"hold": h, "aug": a, "p_delta_gt0": round(p, 4),
                     "checks": checks, "verdict": verdict}
    out["baseline"] = {"hold": base_hold, "aug": base_aug}
    os.makedirs(os.path.join(BASE, "selection_validator", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(BASE, "selection_validator", "results",
                                     "ideas_batch.json"), "w"), indent=1)
    print("\nresults -> selection_validator/results/ideas_batch.json")
