"""mechanics_audit.py — cycle 5: selection-mechanics audit (spec 05).

Dataset-only measurement of the live selection mechanics:
  A) dual-fire census — how often strategies co-fire on the same bar
     (same symbol+ts, both in-band at the LIVE floor/ceil 0.40-0.65),
     direction agreement, and the max-proba pick's performance vs single-fire;
  B) active-lane audit — per symbol x lane full + hold-out stats vs symbol
     and book averages, with a pre-registered removal rule (memo-gated).

FR-003: eligibility mirrors the LIVE mechanism (floor 0.40 / ceil 0.65,
window-filtered, not jump-skipped). Outcome data exists only for rows taken
under the OLD band (0.35-0.50), so outcome-based stats cover that realized
subset — documented, not hidden. No replay, no GPU, no live changes.
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
HOLD_START = pd.Timestamp("2025-11-01", tz="UTC")     # cycle-3 continuity
LIVE_FLOOR, LIVE_CEIL = 0.40, 0.65                    # current live config
MIN_REMOVE_N = 30


def load() -> pd.DataFrame:
    from selection_validator.dataset import load_rows
    from selection_validator.dedup_signals import main as dedup
    from selection_validator.time_window import window_mask
    dedup()
    df = load_rows(*glob.glob(DATA_GLOB))
    m = df["take"].astype(bool) & ~df["jump"].astype(bool)
    df = df[m].reset_index(drop=True)
    df["eligible"] = ((df["proba"] >= LIVE_FLOOR) & (df["proba"] <= LIVE_CEIL)
                      & window_mask(df))
    return df


def census(df: pd.DataFrame) -> dict:
    """Dual-fire census (US1): same symbol+ts, both eligible."""
    elig = df[df["eligible"]]
    counts = elig.groupby(["symbol", "ts"]).size()
    dual = counts[counts >= 2].index
    dual_df = elig.set_index(["symbol", "ts"]).loc[dual].reset_index()
    # direction agreement among dual-fire signals
    dirs = dual_df.groupby(["symbol", "ts"])["direction"].nunique()
    agree = float((dirs == 1).mean()) if len(dirs) else 0.0
    # the max-proba pick per event (the live rule), with realized outcome
    picks = dual_df.sort_values("proba", ascending=False) \
        .groupby(["symbol", "ts"]).head(1)
    p_hit = picks[picks["outcome_r"].notna()]
    picked = {"n": len(p_hit), "wr": float((p_hit["outcome_r"] > 0).mean())
              if len(p_hit) else None,
              "avg_r": float(p_hit["outcome_r"].mean()) if len(p_hit) else None}
    # single-fire reference: eligible events where only one strategy fired
    single = elig.set_index(["symbol", "ts"]).loc[counts[counts == 1].index] \
        .reset_index()
    s_hit = single[single["outcome_r"].notna()]
    single_stats = {"n": len(s_hit), "wr": float((s_hit["outcome_r"] > 0).mean())
                    if len(s_hit) else None,
                    "avg_r": float(s_hit["outcome_r"].mean()) if len(s_hit) else None}
    return {
        "dual_fire_events": int(len(dual)),
        "fraction_of_eligible_events": round(len(dual) / max(len(elig), 1), 4),
        "direction_agreement_rate": round(agree, 4),
        "max_proba_pick": picked,
        "single_fire": single_stats,
    }


def lane_audit(df: pd.DataFrame) -> pd.DataFrame:
    """US2: per symbol x lane full + hold-out stats vs symbol average."""
    from selection_validator.selectors import stats
    rows = []
    for (sym, lane), g in df[df["eligible"] & df["outcome_r"].notna()] \
            .groupby(["symbol", "strategy"]):
        h = g[g["ts"] >= HOLD_START]
        s_full, s_hold = stats(g["outcome_r"]), stats(h["outcome_r"])
        sym_full = stats(df[(df["symbol"] == sym) & df["eligible"]
                            & df["outcome_r"].notna()]["outcome_r"])
        rows.append({
            "symbol": sym, "lane": lane,
            "full_n": s_full["n"], "full_wr": s_full["wr"],
            "full_avg_r": s_full["avg_r"], "full_pf": s_full["pf"],
            "hold_n": s_hold["n"], "hold_wr": s_hold["wr"],
            "hold_avg_r": s_hold["avg_r"], "hold_pf": s_hold["pf"],
            "symbol_avg_r": sym_full["avg_r"],
        })
    return pd.DataFrame(rows)


def removal_flags(audit: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """US3: flag iff hold avgR < 0 AND n>=30 AND P(avgR < symbol avgR) > 0.95."""
    from selection_validator import harness
    flags = []
    for _, r in audit.iterrows():
        if r["hold_n"] < MIN_REMOVE_N or r["hold_avg_r"] is None or r["hold_avg_r"] >= 0:
            flags.append({**r.to_dict(), "flag": False, "why": "no flag"})
            continue
        g = df[(df["symbol"] == r["symbol"]) & df["eligible"]
               & df["outcome_r"].notna()]
        h = g[g["ts"] >= HOLD_START]
        lane_rs = h.loc[h["strategy"] == r["lane"], "outcome_r"]
        other_rs = h.loc[h["strategy"] != r["lane"], "outcome_r"]
        # bootstrap_diff(other, lane) = P(mean(lane) - mean(other) < 0)
        # i.e. P(the lane is WORSE than its symbol's other lanes)
        p_worse = harness.bootstrap_diff(other_rs, lane_rs)
        flag = p_worse > 0.95
        flags.append({**r.to_dict(), "flag": bool(flag),
                      "why": (f"hold avgR {r['hold_avg_r']:.3f} < 0, n={r['hold_n']}, "
                              f"P(lane worse than symbol) = {p_worse:.3f} "
                              f"({'FLAG' if flag else 'not significant'})")})
    return pd.DataFrame(flags)


if __name__ == "__main__":
    import json
    df = load()
    print(f"rows: {len(df)} | eligible (live band 0.40-0.65 + window): "
          f"{int(df['eligible'].sum())}", flush=True)
    c = census(df)
    print("CENSUS:", json.dumps(c, indent=1), flush=True)
    audit = lane_audit(df)
    print("\nLANE AUDIT (eligible, realized outcomes):", flush=True)
    print(audit.to_string(index=False), flush=True)
    flags = removal_flags(audit, df)
    flagged = flags[flags["flag"]]
    print(f"\nREMOVAL FLAGS: {len(flagged)}", flush=True)
    for _, r in flagged.iterrows():
        print(f"  {r['symbol']} {r['lane']}: {r['why']}", flush=True)
    c["lane_audit"] = audit.to_dict("records")
    c["removal_flags"] = flagged.to_dict("records")
    os.makedirs(os.path.join(BASE, "selection_validator", "results"), exist_ok=True)
    json.dump(c, open(os.path.join(BASE, "selection_validator", "results",
                                   "mechanics_audit.json"), "w"), indent=1)
    print("\nresults -> selection_validator/results/mechanics_audit.json")
