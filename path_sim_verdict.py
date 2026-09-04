#!/usr/bin/env python3
"""path_sim_verdict.py — score the pre-registered gates on the pathsim output.
Gates locked in path_sim_test.py docstring BEFORE generation. No tuning."""
import json

import numpy as np
import pandas as pd
from scipy import stats as st

R = json.load(open(os.path.join(os.path.expanduser("~"), ".autotrade_pathsim.json")))
df = pd.DataFrame(R).sort_values("time").reset_index(drop=True)
print(f"scored signals: {len(df)}  |  base WR {100*(df.r>0).mean():.1f}%  "
      f"meanR {df.r.mean():+.3f}")

# chronological split: older 2/3 analysis, newest 1/3 holdout
cut = int(len(df) * 2 / 3)
ana, hold = df.iloc[:cut], df.iloc[cut:]
print(f"analysis: n={len(ana)} ({ana.time.iloc[0][:10]}..{ana.time.iloc[-1][:10]})")
print(f"holdout : n={len(hold)} ({hold.time.iloc[0][:10]}..{hold.time.iloc[-1][:10]})")

# G1 — rank correlation on analysis set
rho, pv = st.spearmanr(ana.p_win, ana.r)
g1 = rho > 0.15
print(f"\nG1 Spearman(analysis) = {rho:+.3f} (p={pv:.4f})  -> {'PASS' if g1 else 'FAIL'}")

# G2 — holdout separation top vs bottom half by p_win
med = ana.p_win.median()          # threshold chosen on ANALYSIS set only
hi, lo = hold[hold.p_win >= med], hold[hold.p_win < med]
wr_hi, wr_lo = (hi.r > 0).mean(), (lo.r > 0).mean()
mr_hi, mr_lo = hi.r.mean(), lo.r.mean()
g2 = (wr_hi - wr_lo >= 0.08) and (mr_hi - mr_lo >= 0.25)
print(f"G2 holdout split @p_win>={med:.3f}:")
print(f"   top   n={len(hi):3d} WR={100*wr_hi:.1f}% meanR={mr_hi:+.3f}")
print(f"   bot   n={len(lo):3d} WR={100*wr_lo:.1f}% meanR={mr_lo:+.3f}")
print(f"   gaps: WR {100*(wr_hi-wr_lo):+.1f}pts (need>=8), meanR {mr_hi-mr_lo:+.3f} "
      f"(need>=+0.25)  -> {'PASS' if g2 else 'FAIL'}")

# G3 — bootstrap the gated holdout subset
rng = np.random.default_rng(42)
kept = hi.r.to_numpy()
p_le0 = None
if len(kept) >= 5:
    boots = [rng.choice(kept, len(kept), replace=True).mean() for _ in range(10000)]
    p_le0 = float(np.mean(np.array(boots) <= 0))
    g3 = p_le0 < 0.20 and len(kept) >= 30
    print(f"\nG3 gate subset: n={len(kept)} meanR={kept.mean():+.3f} "
          f"p(P(meanR<=0))={p_le0:.3f} (need<0.20, n>=30) -> {'PASS' if g3 else 'FAIL'}")
else:
    g3 = False
    print(f"\nG3 gate subset: n={len(kept)} too small -> FAIL")

# G4 — sanity vs grader's own numbers
c_proba = np.corrcoef(df.p_win, df.proba)[0, 1]
c_rhat = np.corrcoef(df.p_win, df.r_hat)[0, 1]
g4 = abs(c_proba) < 0.7 and abs(c_rhat) < 0.7
print(f"\nG4 corr(p_win,proba)={c_proba:+.3f}, corr(p_win,r_hat)={c_rhat:+.3f} "
      f"(both need |.|<0.7) -> {'PASS' if g4 else 'FAIL'}")

verdict = all([g1, g2, g3, g4])
print("\n" + "=" * 60)
print(f"VERDICT: {'PASS — wire as advisory gate after paper-trade' if verdict else 'DEAD — no re-runs with tweaks'}")
print("=" * 60)

json.dump({"gates": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)},
           "spearman": float(rho), "threshold_from_analysis": float(med),
           "holdout_top": {"n": int(len(hi)), "wr": float(wr_hi), "mean_r": float(mr_hi)},
           "holdout_bottom": {"n": int(len(lo)), "wr": float(wr_lo), "mean_r": float(mr_lo)},
           "gate_p_le0": p_le0,
           "corr_proba": float(c_proba), "corr_r_hat": float(c_rhat),
           "verdict": "PASS" if verdict else "DEAD"},
          open(os.path.join(os.path.expanduser("~"), ".autotrade_pathsim_verdict.json"), "w"), indent=2)
