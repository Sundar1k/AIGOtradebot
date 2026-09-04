"""direction_audit.py — cycle 6: regime-conditioned direction (spec 06).

Answers the last direction question with numbers: under which pre-registered
point-in-time conditions does the forward K-bar direction beat 0.50, robustly
across walk-forward folds and on the hold-out — AND does trading the
condition improve the funnel (the book test)?

8 conditions x 2 horizons (K=10, 30) = 16 candidates, all reported; the
primary survivor alone decides. Conditions are FIXED in the spec (FR-002),
computed point-in-time, no look-ahead. No replay, no GPU.
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
HORIZONS = (10, 30)
HIT_BAR = 0.53
FOLD_FRAC = 0.7
MIN_FOLD_N = 150


def load() -> pd.DataFrame:
    from selection_validator.dataset import load_rows
    from selection_validator.dedup_signals import main as dedup
    dedup()
    df = load_rows(*glob.glob(DATA_GLOB))
    m = df["take"].astype(bool) & ~df["jump"].astype(bool)
    return df[m].reset_index(drop=True)


# ── forward returns + condition features (point-in-time, cached) ──────────
def add_forward_and_conditions(df: pd.DataFrame) -> pd.DataFrame:
    bars: dict[str, pd.DataFrame] = {}
    closes: dict[str, pd.Series] = {}

    def _bars(sym):
        if sym not in bars:
            path = os.path.join(BASE, "data", f"{sym}_3min.csv")
            b = pd.read_csv(path).rename(columns={"datetime": "time"})
            b["time"] = pd.to_datetime(b["time"], utc=True)
            bars[sym] = b.reset_index(drop=True)
            closes[sym] = b["close"]
        return bars[sym]

    fwd = {k: [None] * len(df) for k in HORIZONS}
    ema_up = [None] * len(df)
    rsi = [None] * len(df)
    adx_hi = [None] * len(df)
    vol_hi = [None] * len(df)
    mom_up = [None] * len(df)

    from selection_validator.time_window import window_mask
    win = window_mask(df).tolist()

    for i, row in df.iterrows():
        sym, ts = row["symbol"], row["ts"]
        b = _bars(sym)
        idx = b.index[b["time"] <= ts]
        if len(idx) == 0:
            continue
        j = int(idx[-1])
        cl = closes[sym]
        for k in HORIZONS:
            if j + k < len(cl):
                fwd[k][i] = 1 if cl.iloc[j + k] > cl.iloc[j] else (-1 if cl.iloc[j + k] < cl.iloc[j] else 0)
        w = b.iloc[max(0, j - 200): j + 1]
        if len(w) < 40:
            continue
        c = w["close"]
        e10, e30 = c.ewm(span=10, adjust=False).mean().iloc[-1], c.ewm(span=30, adjust=False).mean().iloc[-1]
        ema_up[i] = e10 > e30
        # RSI(14)
        d = c.diff()
        up, dn = d.clip(lower=0), -d.clip(upper=0)
        rs = up.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1] / max(dn.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1], 1e-12)
        rsi[i] = 100 - 100 / (1 + rs)
        # ADX(14) simple approximation via +DI/-DI
        hi, lo = w["high"], w["low"]
        pdm = (hi.diff().clip(lower=0)); ndm = (-lo.diff().clip(lower=0))
        tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
        pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1] / max(atr, 1e-12)
        ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1] / max(atr, 1e-12)
        adx_hi[i] = (abs(pdi - ndi) / max(pdi + ndi, 1e-12)) >= 0.18
        # vol ratio ATR14/ATR100
        tr_s = tr.rolling(14).mean().iloc[-1]
        tr_l = tr.rolling(100).mean().iloc[-1]
        vol_hi[i] = (tr_s / tr_l) >= 1.0 if tr_l and tr_l > 0 else None
        mom_up[i] = c.iloc[-1] > c.iloc[-6] if len(c) > 6 else None

    df["fwd10"] = fwd[10]; df["fwd30"] = fwd[30]
    df["ema_up"] = ema_up; df["rsi"] = rsi; df["adx_hi"] = adx_hi
    df["vol_hi"] = vol_hi; df["mom_up"] = mom_up; df["in_window"] = win
    df["rhat_up"] = (df["r_hat"] > 0).tolist()
    df["proba_lo"] = (df["proba"] < 0.50).tolist()
    return df


# ── condition -> (active mask, predicted direction series) ────────────────
def condition_pred(df: pd.DataFrame, name: str):
    """Returns (active_mask, pred_direction_series) for a condition."""
    if name == "ema_alignment":        # continuation
        return df["ema_up"].notna(), (df["ema_up"].astype(float) * 2 - 1).where(df["ema_up"].notna())
    if name == "rsi_meanrev":          # mean reversion
        lo = df["rsi"] < 30; hi = df["rsi"] > 70
        act = lo | hi
        pred = pd.Series(np.where(lo, 1, np.where(hi, -1, 0)), index=df.index)
        return act, pred
    if name == "rhat_sign":            # model expected-R direction
        act = df["r_hat"].notna() & (df["r_hat"] != 0)
        return act, (df["rhat_up"].astype(float) * 2 - 1).where(act)
    # momentum-based (continuation), conditioned on the filter state
    base = df["mom_up"].notna()
    pred = (df["mom_up"].astype(float) * 2 - 1).where(base)
    if name == "momentum":
        return base, pred
    if name == "adx_hi":
        return base & df["adx_hi"].notna() & df["adx_hi"], pred.where(base & df["adx_hi"].notna())
    if name == "adx_lo":
        return base & df["adx_hi"].notna() & ~df["adx_hi"], pred.where(base & df["adx_hi"].notna())
    if name == "flow_vol":
        return base & df["vol_hi"].notna() & ~df["vol_hi"], pred.where(base & df["vol_hi"].notna())
    if name == "chop_vol":
        return base & df["vol_hi"].notna() & df["vol_hi"], pred.where(base & df["vol_hi"].notna())
    if name == "in_window":
        return base & df["in_window"], pred.where(base & df["in_window"])
    if name == "out_window":
        return base & ~df["in_window"], pred.where(base & ~df["in_window"])
    if name == "proba_lo":
        return base & df["proba_lo"], pred.where(base & df["proba_lo"])
    if name == "proba_hi":
        return base & ~df["proba_lo"], pred.where(base & ~df["proba_lo"])
    raise ValueError(name)


CONDITIONS = ["ema_alignment", "rsi_meanrev", "rhat_sign", "momentum",
              "adx_hi", "adx_lo", "flow_vol", "chop_vol",
              "in_window", "out_window", "proba_lo", "proba_hi"]


def hit_rate(df: pd.DataFrame, cond: str, k: int) -> dict:
    act, pred = condition_pred(df, cond)
    fwd = df[f"fwd{k}"]
    ok = act & pred.notna() & fwd.notna() & (fwd != 0)
    hits = (pred[ok] == fwd[ok])
    n = int(hits.sum()) if False else int(ok.sum())
    return {"n": n, "hit": float(hits.mean()) if n else None}


def bootstrap_p_gt_half(hits: np.ndarray, draws=10000, seed=42) -> float:
    """P(bootstrap mean hit-rate > 0.50) — Bernoulli, fixed seed."""
    if hits.size < 30:
        return 0.5
    rng = np.random.default_rng(seed)
    means = rng.choice(hits, size=(draws, hits.size), replace=True).mean(axis=1)
    return float((means > 0.5).mean())


if __name__ == "__main__":
    df = load()
    print(f"signals: {len(df)} | building forward returns + conditions...", flush=True)
    df = add_forward_and_conditions(df)
    from selection_validator.evolve_search import make_folds
    folds = make_folds(df[df["ts"] < HOLD_START])
    hold = df[df["ts"] >= HOLD_START]

    results = {}
    for cond in CONDITIONS:
        for k in HORIZONS:
            fold_hits = []
            for _, fd in folds:
                r = hit_rate(fd, cond, k)
                fold_hits.append(r)
            valid = [r for r in fold_hits if r["n"] >= MIN_FOLD_N]
            ok = [r for r in valid if r["hit"] is not None and r["hit"] >= HIT_BAR]
            frac = len(ok) / max(len(fold_hits), 1)
            med = float(np.median([r["hit"] for r in valid if r["hit"] is not None])) if valid else None
            results[f"{cond}_k{k}"] = {"fold_frac_ok": round(frac, 3), "median_hit": med,
                                       "fold_n_ok": len(ok), "fold_total": len(fold_hits)}
    print("\nPER-FOLD SURVIVAL (hit>=0.53 in >=70% of folds, n>=150/fold):", flush=True)
    survivors = {k: v for k, v in results.items() if v["fold_frac_ok"] >= FOLD_FRAC}
    for k, v in sorted(results.items(), key=lambda x: -(x[1]["median_hit"] or 0)):
        print(f"  {k:22s} frac_ok={v['fold_frac_ok']:.2f} median={v['median_hit']} "
              f"({'SURVIVES' if k in survivors else ''})", flush=True)
    print(f"\nsurvivors: {list(survivors)}", flush=True)

    verdict = "KILL"
    report = {"survivors": list(survivors), "verdict": None}
    if survivors:
        primary = max(survivors, key=lambda k: survivors[k]["median_hit"] or 0)
        cond, k = primary.rsplit("_k", 1)
        k = int(k)
        act, pred = condition_pred(hold, cond)
        fwd = hold[f"fwd{k}"]
        ok = act & pred.notna() & fwd.notna() & (fwd != 0)
        hits = (pred[ok] == fwd[ok]).astype(int).to_numpy()
        p = bootstrap_p_gt_half(hits)
        n_hold = int(ok.sum())
        # book test: aligned-only vs all funnel trades on hold-out
        from selection_validator.selectors import stats
        aligned = hold[ok & hold["outcome_r"].notna()]
        base_t = hold[hold["outcome_r"].notna()]
        aligned_rs = aligned["outcome_r"]
        a = stats(aligned_rs) if len(aligned) else {"n": 0, "avg_r": None, "pf": None}
        b = stats(base_t["outcome_r"])
        aug = df[df["ts"] >= pd.Timestamp("2026-08-01", tz="UTC")]
        aok = act & pred.notna() & aug["outcome_r"].notna()
        a_aug = stats(aug.loc[aok, "outcome_r"]) if aok.sum() else {"avg_r": None}
        b_aug = stats(aug["outcome_r"].dropna())
        checks = {
            "a_hit_gt_50_p": p > 0.95,
            "b_n_ge_150": n_hold >= 150,
            "c_book_avgR": (a["avg_r"] or 0) > (b["avg_r"] or 0),
            "c_book_PF": (a["pf"] or 0) >= (b["pf"] or 0),
            "d_aug_not_worse": (a_aug.get("avg_r") or -9) > (b_aug.get("avg_r") or -9),
        }
        if all(checks.values()):
            verdict = "GO"
        elif p <= 0.5 or not (checks["c_book_avgR"] and checks["c_book_PF"]):
            verdict = "KILL"
        else:
            verdict = "INCONCLUSIVE"
        report.update({"primary": primary, "p_hit_gt_50": round(p, 4), "n_hold": n_hold,
                       "aligned_book": a, "baseline_book": b,
                       "aug_aligned": a_aug, "aug_base": b_aug, "checks": checks})
    report["verdict"] = verdict
    print(json.dumps(report, indent=1), flush=True)
    os.makedirs(os.path.join(BASE, "selection_validator", "results"), exist_ok=True)
    json.dump({"per_candidate": results, "report": report},
              open(os.path.join(BASE, "selection_validator", "results",
                                "direction_audit.json"), "w"), indent=1)
