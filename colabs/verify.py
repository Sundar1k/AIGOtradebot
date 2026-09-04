#!/usr/bin/env python3
"""colabs/verify.py — VERIFICATION GATE for the reconstructed labelers.

Reproduces each labeler's full-span build() on 3-min data and compares
signal count + label distribution against the SHIPPED bundle metadata.
If the reconstruction is faithful, ema should give 85,158 signals with
label_dist ~[59894, 25264] (2R-before-0.5ATR-stop binary outcome).

Usage: ./.venv/bin/python -m colabs.verify
"""
import json
import warnings
import glob

import joblib
import numpy as np

warnings.filterwarnings("ignore")

from colabs.labelers import (EMACrossChronos, SuperTrendChronos,
                             ORBChronos, GannChronos)


def shipped_meta():
    meta = {}
    for f in sorted(glob.glob("models/*_chronos.joblib")):
        b = joblib.load(f)
        md = b.get("training_metadata") or {}
        meta[b.get("labeler_name")] = (md.get("n_train_signals"),
                                       tuple(md.get("label_dist") or []))
    return meta


def main():
    ship = shipped_meta()
    lab_map = {"EMACrossChronos": EMACrossChronos,
               "SuperTrendChronos": SuperTrendChronos,
               "ORBChronos": ORBChronos,
               "GannChronos": GannChronos}   # no shipped gann bundle yet
    print(f"{'labeler':<22}{'n_sig':>9}{'dist':>20}  vs shipped (n, dist)")
    print("-" * 75)
    ok = True
    for name, cls in lab_map.items():
        lab = cls(tf="3min")
        lo = min(df.index.min() for df in lab._b.values())
        hi = max(df.index.max() for df in lab._b.values())
        ctx, y, keys = lab.build(lo, hi, None)
        n = len(y)
        dist = [int((y == 0).sum()), int((y == 1).sum())]
        sn, sd = ship.get(name, (None, None))
        match = ""
        if sn is not None:
            ok_n = abs(n - sn) / sn < 0.05
            ok_d = sd and abs(dist[1] - sd[1]) / sd[1] < 0.10
            match = "  ✓ MATCH" if (ok_n and ok_d) else "  ✗ MISMATCH"
            ok &= (ok_n and ok_d)
        print(f"{name:<22}{n:>9,}{str(dist):>20}  vs {sn} {sd}{match}")
        # also check feature width
        if len(keys):
            F = lab.features(keys[:50])
            print(f"  feat width: {F.shape[1]} (76 FFM + {F.shape[1]-76} handcrafts)")
    print("-" * 75)
    print("VERDICT:", "PASS — labelers are faithful, proceed to 15-min training"
          if ok else "FAIL — inspect the mismatch before training")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
