import os
#!/usr/bin/env python3
"""outcome_clf.py — PILOT: does a setup-outcome classifier separate winners
from losers on the replay ledger?

PRE-REGISTERED GATES (locked before training, 2026-08-26):
  G1 holdout top-half vs bottom-half (by classifier score):
     WR_top - WR_bot >= 8pts AND meanR_top - meanR_bot >= +0.25R, n_bot >= 30
  G2 bootstrap (10k, seed 42) on top-half holdout R: p(P(meanR<=0)) < 0.20
  G3 AUC >= 0.55 on holdout (weak-but-real discrimination)
  G4 sanity: score correlation with proba < 0.8 (not just repackaging the
     existing confidence)

Caveat per protocol: ledger n=581 is BELOW the >=2k the full experiment
needs. This is a PILOT — wiring into live is ADVISORY (log-only) no matter
what. Verdict recorded; no re-runs with tweaks.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEDGER = os.path.join(os.path.expanduser("~"), ".autotrade_missed.json")


def build_features():
    d = json.load(open(LEDGER))
    recs = d["records"] if isinstance(d, dict) else d
    rows = []
    for r in recs:
        t = pd.Timestamp(r["time"])
        rows.append({
            "symbol": r["symbol"],
            "hour": t.hour,
            "dow": t.dayofweek,
            "proba": r["proba"],
            "r_hat": r["r_hat"],
            "conflict": int(bool(r.get("conflict", False))),
            "n_patterns": len(r.get("patterns", []) or []),
            "pattern_dir": int(r.get("pattern_dir", 0) or 0),
            "dir": int(r.get("dir", 1)),
            "aligned": int((r.get("pattern_dir") or 0) == (r.get("dir") or 0)
                           and len(r.get("patterns", []) or []) > 0),
            "win": int(r["r"] > 0),
            "r": r["r"],
            "time": t,
        })
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df = pd.get_dummies(df, columns=["symbol"], prefix="sym")
    return df


def main():
    df = build_features()
    feat_cols = [c for c in df.columns if c not in ("win", "r", "time")]
    X, y = df[feat_cols], df["win"]
    # chronological split: older 2/3 train, newest 1/3 holdout
    cut = int(len(df) * 2 / 3)
    Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
    Xte, yte, rte = X.iloc[cut:], y.iloc[cut:], df["r"].iloc[cut:]
    t0 = str(df.time.iloc[0])[:10]
    t1 = str(df.time.iloc[cut - 1])[:10]
    t2 = str(df.time.iloc[cut])[:10]
    t3 = str(df.time.iloc[-1])[:10]
    print(f"ledger n={len(df)} | train n={len(Xtr)} ({t0}..{t1}) | "
          f"holdout n={len(Xte)} ({t2}..{t3})")

    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", n_jobs=4,
                      random_state=42)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    df["score"] = np.nan
    df.loc[Xte.index, "score"] = p

    # G1: top vs bottom half of holdout by score
    te = df.iloc[cut:].copy()
    med = float(np.median(p))
    hi = te[te["score"] >= med]
    lo = te[te["score"] < med]
    wr_hi, wr_lo = hi["win"].mean(), lo["win"].mean()
    mr_hi, mr_lo = hi["r"].mean(), lo["r"].mean()
    g1 = (wr_hi - wr_lo >= 0.08) and (mr_hi - mr_lo >= 0.25) and len(lo) >= 30
    print(f"\nG1 top-half (n={len(hi)}): WR {wr_hi:.3f} meanR {mr_hi:+.3f} | "
          f"bot (n={len(lo)}): WR {wr_lo:.3f} meanR {mr_lo:+.3f} | "
          f"gaps WR {wr_hi-wr_lo:+.3f} meanR {mr_hi-mr_lo:+.3f} -> "
          f"{'PASS' if g1 else 'FAIL'}")

    # G2 bootstrap on top-half R
    rng = np.random.default_rng(42)
    boots = [rng.choice(hi["r"].to_numpy(), len(hi), replace=True).mean()
             for _ in range(10000)]
    p_le0 = float(np.mean(np.array(boots) <= 0))
    g2 = p_le0 < 0.20
    print(f"G2 top-half bootstrap p(P(meanR<=0))={p_le0:.3f} -> "
          f"{'PASS' if g2 else 'FAIL'}")

    # G3 AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(yte, p)
    g3 = auc >= 0.55
    print(f"G3 holdout AUC = {auc:.3f} -> {'PASS' if g3 else 'FAIL'}")

    # G4 independence from proba
    c_proba = float(np.corrcoef(p, te["proba"].to_numpy())[0, 1])
    c_rhat = float(np.corrcoef(p, te["r_hat"].to_numpy())[0, 1])
    g4 = abs(c_proba) < 0.8
    print(f"G4 corr(score,proba)={c_proba:+.3f} corr(score,r_hat)={c_rhat:+.3f} "
          f"-> {'PASS' if g4 else 'FAIL'}")

    verdict = all([g1, g2, g3, g4])
    print("\n" + "=" * 60)
    print(f"VERDICT: {'PASS (pilot) — wire ADVISORY log-only' if verdict else 'DEAD — no re-runs'}")
    print("=" * 60)
    imp = sorted(zip(feat_cols, m.feature_importances_),
                 key=lambda t: -t[1])[:8]
    print("\ntop features:", ", ".join(f"{k}={v:.3f}" for k, v in imp))

    out = {"gates": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
                     "G4": bool(g4)}, "auc": auc, "wr_hi": float(wr_hi),
           "wr_lo": float(wr_lo), "mr_hi": float(mr_hi),
           "mr_lo": float(mr_lo), "p_le0": p_le0,
           "corr_proba": c_proba, "corr_r_hat": c_rhat,
           "n_train": int(len(Xtr)), "n_holdout": int(len(Xte)),
           "verdict": "PASS" if verdict else "DEAD"}
    json.dump(out, open(os.path.join(os.path.expanduser("~"), ".autotrade_outcome_clf.json"), "w"),
              indent=2)
    # stash model for advisory wiring
    import joblib
    joblib.dump({"model": m, "feat_cols": feat_cols,
                 "fit_time": str(pd.Timestamp.utcnow())},
                os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/outcome_clf.joblib"))
    print("model saved -> outcome_clf.joblib")


if __name__ == "__main__":
    main()
