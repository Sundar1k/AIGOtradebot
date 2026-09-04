"""evolve_search.py — cycle 4: walk-forward selection-layer search (spec 04).

The user's "try thousands of possibilities" idea, done safely: search the
selection layer (floor x ceil x chop_max grids) on the existing scored-signal
dataset with walk-forward folds; ONE survivor faces the pre-registered
hold-out verdict (ts >= 2025-11-01, cycle-3 boundary). Live money only ever
sees a pre-registered winner. FR-007: detection logic, models, exit and the
09:30-12:00 ET window gate are FROZEN — only the grids below are searched.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
DATA_GLOB = os.path.join(BASE, "selection_validator", "data", "signals_*.jsonl")
TRAIL_BARS = 300
HOLD_START = pd.Timestamp("2025-11-01", tz="UTC")     # cycle-3 continuity (FR-003)

# ── pre-registered grids (spec FR-007) ────────────────────────────────────
FLOORS = [0.25, 0.30, 0.35, 0.40]
CEILS = [0.50, 0.55, 0.60, 0.65]
CHOPS = [0.8, 1.0, 1.2, 1.5, 2.0]
LIVE = (0.35, 0.50, 1.0)          # the current live config (comparison baseline)
PF_FOLD_MIN = 1.8                 # robust-selection bar per fold
PF_FOLD_FRAC = 0.7                # ...in >=70% of folds (FR-004)
MIN_FOLD_N = 30                   # sparse folds don't count toward the PF rule


def load_funnel() -> pd.DataFrame:
    from selection_validator.dataset import load_rows
    from selection_validator.dedup_signals import main as dedup
    dedup()
    df = load_rows(*glob.glob(DATA_GLOB))
    m = df["take"].astype(bool) & ~df["jump"].astype(bool)
    return df[m].reset_index(drop=True)


def cache_chop_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """One ATR14/ATR100 ratio per signal (FR-002) — point-in-time, cached.

    All chop thresholds derive from this single cached ratio; the heavy
    bars-at-ts work happens exactly once per (symbol, ts)."""
    import chop_gate
    bars_cache: dict[str, pd.DataFrame] = {}
    ratios: dict[tuple, float] = {}
    for row in df[["symbol", "ts"]].drop_duplicates().to_dict("records"):
        sym, ts = row["symbol"], row["ts"]
        key = (sym, str(ts))
        try:
            if sym not in bars_cache:
                path = os.path.join(BASE, "data", f"{sym}_3min.csv")
                b = pd.read_csv(path).rename(columns={"datetime": "time"})
                b["time"] = pd.to_datetime(b["time"], utc=True)
                bars_cache[sym] = b
            win = bars_cache[sym][bars_cache[sym]["time"] <= ts].tail(TRAIL_BARS)
            ratios[key] = chop_gate.atr_ratio(win)
        except Exception:
            ratios[key] = float("nan")          # fail-open (live semantics)
    df["atr_ratio"] = df.apply(
        lambda r: ratios.get((r["symbol"], str(r["ts"])), float("nan")), axis=1)
    return df


def candidate_mask(df: pd.DataFrame, floor: float, ceil: float,
                   chop_max: float | None) -> pd.Series:
    from selection_validator.time_window import window_mask
    m = (df["proba"] >= floor) & (df["proba"] <= ceil) & ~df["jump"].astype(bool)
    m &= window_mask(df)
    if chop_max is not None:
        r = df["atr_ratio"]
        m &= r.isna() | (r < chop_max)            # fail-open on NaN
    return m


def evaluate(df: pd.DataFrame, floor: float, ceil: float,
             chop_max: float | None) -> dict:
    from selection_validator.selectors import stats
    mask = candidate_mask(df, floor, ceil, chop_max)
    rs = df.loc[mask & df["outcome_r"].notna(), "outcome_r"]
    out = stats(rs)
    out["signals"] = int(mask.sum())
    return out


def make_folds(train: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Contiguous ~6-month folds over pre-hold data (FR-003, no overlap)."""
    edges = pd.date_range(start=pd.Timestamp("2021-04-01", tz="UTC"),
                          end=HOLD_START, freq="6MS")
    folds = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        folds.append((f"{a.date()}..{b.date()}", train[(train['ts'] >= a) & (train['ts'] < b)]))
    folds.append((f"{edges[-1].date()}..2025-10-31",
                  train[(train['ts'] >= edges[-1]) & (train['ts'] < HOLD_START)]))
    return folds


def fold_stats(fold: pd.DataFrame, floor: float, ceil: float,
               chop_max: float | None) -> dict:
    e = evaluate(fold, floor, ceil, chop_max)
    return {"closed": e["n"], "avg_r": e["avg_r"], "pf": e["pf"], "sum_r": e["sum_r"]}


def select_robust(fold_tables: dict, candidates: list) -> tuple:
    """FR-004: best median fold avgR among PF>=1.8 in >=70% of folds.

    Sparse/empty folds count as failed toward the 70% rule. Tiebreaks:
    higher median PF, fewer negative folds, grid order."""
    ranked = []
    for cfg in candidates:
        rows = fold_tables[cfg]
        valid = [r for r in rows if r["closed"] >= MIN_FOLD_N]
        ok = [r for r in valid if r["pf"] is not None and r["pf"] >= PF_FOLD_MIN]
        frac = len(ok) / max(len(rows), 1)
        if frac < PF_FOLD_FRAC:
            continue
        med_avg = float(np.median([r["avg_r"] for r in valid]))
        med_pf = float(np.median([r["pf"] for r in valid if r["pf"] is not None] or [0]))
        neg_folds = sum(1 for r in valid if r["avg_r"] is not None and r["avg_r"] < 0)
        ranked.append((med_avg, med_pf, -neg_folds, cfg))
    ranked.sort(reverse=True)
    return [c for _, _, _, c in ranked]


def decide_verdict(cand: dict, live: dict, n_hold: int, aug_cand: dict,
                   aug_live: dict, p_delta_gt0: float) -> tuple[str, dict]:
    checks = {
        "a_p_gt_095": p_delta_gt0 > 0.95,
        "b_pf_cand_ge_live": (cand["pf"] or 0) >= (live["pf"] or 0),
        "c_aug_not_worse": aug_cand["avg_r"] is not None and aug_live["avg_r"] is not None
            and aug_cand["avg_r"] > aug_live["avg_r"],
        "d_n_cand_ge_150": (cand["n"] or 0) >= 150,
    }
    if all(checks.values()):
        return "GO", checks
    if p_delta_gt0 <= 0.5 or (cand["pf"] or 0) < (live["pf"] or 0) or not checks["c_aug_not_worse"]:
        return "KILL", checks
    return "INCONCLUSIVE", checks


if __name__ == "__main__":
    from selection_validator import harness
    from selection_validator.time_window import window_mask

    df = load_funnel()
    print(f"funnel rows: {len(df)} | caching chop ratios...", flush=True)
    df = cache_chop_ratios(df)
    print("ratios cached", flush=True)

    # sanity (SC-001): chop-off window config reproduces cycle-3 full window
    san = evaluate(df, 0.35, 0.50, None)
    ok = san["avg_r"] is not None and 0.60 <= san["avg_r"] <= 0.68 and 2.2 <= (san["pf"] or 0) <= 2.7
    print(f"SANITY: window(0.35,0.50,chop-off) avgR={san['avg_r']} PF={san['pf']} "
          f"({'PASS' if ok else 'FAIL — stop'})", flush=True)
    if not ok:
        sys.exit(1)

    train = df[df["ts"] < HOLD_START]
    hold = df[df["ts"] >= HOLD_START]
    aug = df[df["ts"] >= pd.Timestamp("2026-08-01", tz="UTC")]
    folds = make_folds(train)
    print(f"folds: {len(folds)} | train rows {len(train)} | hold rows {len(hold)}", flush=True)

    candidates = [(f, c, ch) for f in FLOORS for c in CEILS for ch in CHOPS
                  if (f, c, ch) != LIVE]
    fold_tables = {}
    for cfg in candidates:
        fold_tables[cfg] = [fold_stats(fd, *cfg) for _, fd in folds]
    live_folds = [fold_stats(fd, *LIVE) for _, fd in folds]
    print(f"evaluated {len(candidates)} candidates x {len(folds)} folds", flush=True)

    top = select_robust(fold_tables, candidates)[:3]
    print("top-3:", [(f, c, ch) for f, c, ch in top], flush=True)
    if not top:
        print("NO candidate passed robust selection — verdict KILL (live config stays)")
        sys.exit(0)

    best = top[0]
    live_hold = evaluate(hold, *LIVE)
    cand_hold = evaluate(hold, *best)
    live_aug = evaluate(aug, *LIVE)
    cand_aug = evaluate(aug, *best)
    lm = candidate_mask(hold, *LIVE)
    cm = candidate_mask(hold, *best)
    p_lt0 = harness.bootstrap_diff(hold.loc[lm & hold["outcome_r"].notna(), "outcome_r"],
                                   hold.loc[cm & hold["outcome_r"].notna(), "outcome_r"])
    p_delta = round(1 - p_lt0, 4)
    verdict, checks = decide_verdict(cand_hold, live_hold, len(hold),
                                     cand_aug, live_aug, p_delta)
    print(json_dump := __import__("json").dumps({
        "verdict": verdict, "checks": checks, "best": list(best),
        "hold_live": live_hold, "hold_cand": cand_hold,
        "aug_live": live_aug, "aug_cand": cand_aug, "p_delta_gt0": p_delta}, indent=1))
