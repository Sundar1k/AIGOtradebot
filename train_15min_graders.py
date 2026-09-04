#!/usr/bin/env python3
"""train_15min_graders.py — Phase C: train 15-min lane graders.

Drives futures_foundation.pipeline.produce.train() (the SAME production
trainer that shipped the 3-min bundles) for the ACTIVE lanes on 15-min
bars, with the `_15min` model_filename convention config.py requires
(no cross-timeframe fallback).

Lanes: ema (5 handcrafts), supertrend (2), orb (7), gann (2 — mirrors
supertrend, first ML bundle for the Gann lane).

Holdout: last 4 months of the 15-min data (OOS sanity, same as the
walk-forward's clean-holdout convention). Outputs go to models/ with the
exact naming the strategy model_path() expects when TIMEFRAME_MIN=15:
  ema_cross_chronos_15min.joblib
  supertrend_chronos_15min.joblib
  orb_chronos_15min.joblib
  gann_chronos_15min.joblib

Run (CPU, safe during live trading — no GPU, no veto dependency):
  ./.venv/bin/python train_15min_graders.py            # all lanes
  ./.venv/bin/python train_15min_graders.py --lane ema # one lane
"""
import argparse
import os
import sys
import time

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LANES = {
    "ema":       ("colabs.labelers:EMACrossChronos",   "ema_cross_chronos_15min.joblib"),
    "supertrend": ("colabs.labelers:SuperTrendChronos", "supertrend_chronos_15min.joblib"),
    "orb":       ("colabs.labelers:ORBChronos",        "orb_chronos_15min.joblib"),
    "gann":      ("colabs.labelers:GannChronos",       "gann_chronos_15min.joblib"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default=None, help="one lane (ema|supertrend|orb|gann); default all")
    ap.add_argument("--holdout-months", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lanes = [args.lane] if args.lane else list(LANES)
    results = {}
    for lane in lanes:
        spec, fname = LANES[lane]
        mod_name, cls_name = spec.split(":")
        import importlib
        cls = getattr(importlib.import_module(mod_name), cls_name)
        lab = cls(tf="15min")
        out = os.path.join("models", fname)
        print(f"\n{'#'*70}\n# TRAIN {lane} → {out}\n{'#'*70}", flush=True)
        t0 = time.time()
        from futures_foundation.pipeline import produce
        meta = produce.train(
            lab, holdout_months=args.holdout_months, seed=args.seed,
            output_path=out, verbose=True)
        dt = time.time() - t0
        tm = meta.get("training_metadata", {})
        he = meta.get("holdout_eval", {})
        results[lane] = dict(
            file=out, n_train=tm.get("n_train_signals"),
            dist=tm.get("label_dist"),
            holdout_n=he.get("n_signals"),
            holdout_pos=he.get("positive_rate"),
            fixed_tp=he.get("fixed_tp_by_thr"),
            train_secs=round(dt, 1))
        print(f"\n✅ {lane} done in {dt/60:.1f} min → {out}", flush=True)
        print(f"   n_train={results[lane]['n_train']}  dist={results[lane]['dist']}")
        print(f"   holdout n={results[lane]['holdout_n']}  pos_rate={results[lane]['holdout_pos']:.3f}")
        if he.get("fixed_tp_by_thr"):
            print("   holdout fixed-TP sweep (thr: trades/WR/meanR/PF):")
            for thr, r in sorted(he["fixed_tp_by_thr"].items()):
                if r and r.get("trades"):
                    print(f"     {thr}: {r['trades']} / {100*r['wr']:.1f}% / {r['meanR']:+.2f}R / PF {r['pf']:.2f}")

    print("\n" + "=" * 70)
    print("PHASE C SUMMARY")
    for lane, r in results.items():
        print(f"  {lane:<12} {r['file']:<40} n={r['n_train']}  holdout_n={r['holdout_n']}  {r['train_secs']}s")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
