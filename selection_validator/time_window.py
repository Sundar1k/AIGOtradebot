"""time_window.py — cycle 3: 09:30-12:00 ET window gate vs all-day (spec 03).

The user's hypothesis: the bot's edge concentrates in the US cash-open
through noon (09:30-12:00 ET). Measured on history: +0.660R / PF 2.54 vs
+0.502 / 2.01 all-day. This module runs the PRE-REGISTERED blind test:
window vs all-day on the fixed OOS slice (ts >= 2025-11-01, FR-004), with
the four-condition GO rule and a non-deciding stability report.

Zero free parameters — the gate is a pure time filter. Live config/loop
are never touched (FR-006).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_GLOB = os.path.join(BASE, "selection_validator", "data", "signals_*.jsonl")
OOS_BOUNDARY = "2025-11-01"          # FR-004 — fixed at spec time
WIN_LO, WIN_HI = 570, 720            # 09:30-12:00 ET in minutes (half-open)
BOUNDARY = pd.Timestamp(OOS_BOUNDARY, tz="UTC")


def et_minute(ts) -> pd.Series:
    """ET minute-of-day via America/New_York wall time — DST-correct.

    A fixed UTC offset would misclassify the window by an hour for the ~5
    winter months per year (EST vs EDT). The dataset spans both; the live
    gate must use the same TZ-aware logic (spec edge case, DST).
    """
    et = ts.dt.tz_convert("America/New_York")
    return et.dt.hour * 60 + et.dt.minute


def window_mask(df: pd.DataFrame) -> pd.Series:
    m = et_minute(df["ts"])
    return (m >= WIN_LO) & (m < WIN_HI)


def load_funnel() -> pd.DataFrame:
    from selection_validator.dataset import load_rows
    df = load_rows(*glob.glob(DATA_GLOB))
    m = df["take"].astype(bool) & ~df["jump"].astype(bool)
    return df[m].reset_index(drop=True)


def compare(df: pd.DataFrame, oos: pd.DataFrame, aug: pd.DataFrame):
    """window vs all-day stats + bootstrap P(ΔavgR>0) per slice."""
    from selection_validator.selectors import BaselineSelector, evaluate
    from selection_validator import harness
    wm = window_mask(df)
    base_all = evaluate(df, BaselineSelector())
    win_all = evaluate(df[wm], BaselineSelector())
    base_oos = evaluate(oos, BaselineSelector())
    win_oos = evaluate(oos[window_mask(oos)], BaselineSelector())
    base_aug = evaluate(aug, BaselineSelector())
    win_aug = evaluate(aug[window_mask(aug)], BaselineSelector())
    # Condition (a) per spec: bootstrap on the OOS SLICE trades only (FR-004).
    p_lt0 = harness.bootstrap_diff(oos["outcome_r"].dropna(),
                                   oos[window_mask(oos)]["outcome_r"].dropna())
    return {
        "full": {"baseline": base_all, "window": win_all},
        "oos": {"baseline": base_oos, "window": win_oos,
                "p_delta_gt0": round(1 - p_lt0, 4)},
        "august": {"baseline": base_aug, "window": win_aug},
    }


def decide_verdict(comp: dict) -> tuple[str, dict]:
    o, a = comp["oos"], comp["august"]
    b, w = o["baseline"], o["window"]
    p = o["p_delta_gt0"]
    checks = {
        "a_p_gt_095": p > 0.95,
        "b_pf_win_ge_base": (w["pf"] or 0) >= (b["pf"] or 0),
        "c_aug_not_worse": a["window"]["avg_r"] is not None and a["baseline"]["avg_r"] is not None
            and a["window"]["avg_r"] > a["baseline"]["avg_r"],
        "d_n_win_oos_ge_100": (w["closed"] or 0) >= 100,
    }
    if all(checks.values()):
        return "GO", checks
    if p <= 0.5 or (w["pf"] or 0) < (b["pf"] or 0) or not checks["c_aug_not_worse"]:
        return "KILL", checks
    if (w["closed"] or 0) < 100:
        return "INCONCLUSIVE", checks
    return "INCONCLUSIVE", checks


def stability(df: pd.DataFrame) -> dict:
    """Per-year + per-symbol window-vs-all-day (report only, no decision power)."""
    from selection_validator.selectors import BaselineSelector, evaluate, stats
    out = {"by_year": {}, "by_symbol": {}}
    for y in sorted(df["ts"].dt.year.unique()):
        g = df[df["ts"].dt.year == y]
        out["by_year"][str(y)] = {
            "all": stats(g["outcome_r"]), "window": stats(g[window_mask(g)]["outcome_r"])}
    for sym in sorted(df["symbol"].unique()):
        g = df[df["symbol"] == sym]
        out["by_symbol"][sym] = {
            "all": stats(g["outcome_r"]), "window": stats(g[window_mask(g)]["outcome_r"])}
    return out


if __name__ == "__main__":
    from selection_validator.dedup_signals import main as dedup
    dedup()
    df = load_funnel()
    oos = df[df["ts"] >= BOUNDARY]
    aug = df[df["ts"] >= pd.Timestamp("2026-08-01", tz="UTC")]
    comp = compare(df, oos, aug)
    verdict, checks = decide_verdict(comp)
    print(json.dumps({"verdict": verdict, "checks": checks}, indent=1))
    print("OOS baseline:", json.dumps(comp["oos"]["baseline"]))
    print("OOS window:  ", json.dumps(comp["oos"]["window"]))
    print("AUG baseline:", json.dumps(comp["august"]["baseline"]))
    print("AUG window:  ", json.dumps(comp["august"]["window"]))
    print("FULL baseline:", json.dumps(comp["full"]["baseline"]))
    print("FULL window:  ", json.dumps(comp["full"]["window"]))
    st = stability(df)
    os.makedirs(os.path.join(BASE, "selection_validator", "results"), exist_ok=True)
    json.dump(st, open(os.path.join(BASE, "selection_validator", "results",
                                    "time_window_stability.json"), "w"), indent=1)
    print("stability written")
