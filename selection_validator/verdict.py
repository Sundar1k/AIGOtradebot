"""verdict.py — numbers-backed verdict report (FR-004, US4).

Writes specs/selection-validation/verdict.md. Numbers only for the decision;
prose is limited to interpretation of those numbers.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import pandas as pd

from selection_validator import harness
from selection_validator.dataset import OOS_BOUNDARY, leak_check, load_rows
from selection_validator.selectors import BaselineSelector, CandidateSelector, evaluate

VERDICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "specs", "selection-validation", "verdict.md",
)
DATASET_GLOB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "selection_validator", "data", "signals_*.jsonl",
)


def run_verdict(df: pd.DataFrame, q: int, boundary: str = OOS_BOUNDARY,
                verdict_path: str = VERDICT_PATH) -> dict:
    """Full pre-registered verdict for one candidate q on the OOS slice."""
    train: pd.DataFrame
    oos: pd.DataFrame
    train, oos = harness.split_slices(df, boundary)

    base_train = evaluate(train, BaselineSelector())
    base_oos = evaluate(oos, BaselineSelector())
    cand = CandidateSelector(q)
    cand_train = evaluate(train, cand)
    cand_oos = evaluate(oos, cand)

    # Bootstrap on OOS closed trades only (one code path, FR-002).
    base_mask = BaselineSelector().accept(oos) & oos["outcome_r"].notna()
    cand_mask = cand.accept(oos) & oos["outcome_r"].notna()
    base_rs = oos.loc[base_mask, "outcome_r"]
    cand_rs = oos.loc[cand_mask, "outcome_r"]
    p_lt0 = harness.bootstrap_diff(base_rs, cand_rs)
    cand_oos["p_delta_lt0"] = p_lt0

    n_oos = int(oos.shape[0])
    decision, info = harness.decide(base_oos, cand_oos, n_oos)

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "boundary": boundary,
        "dataset_rows": int(df.shape[0]),
        "q": q,
        "decision": decision,
        "decision_info": info,
        "train": {"baseline": base_train, "candidate": cand_train},
        "oos": {"baseline": base_oos, "candidate": cand_oos,
                "p_delta_lt0": round(p_lt0, 4),
                "p_delta_gt0": round(1 - p_lt0, 4)},
    }
    os.makedirs(os.path.dirname(verdict_path), exist_ok=True)
    with open(verdict_path, "w") as f:
        f.write(_render(report))
    return report


def _fmt(st: dict) -> str:
    pf = st.get("pf")
    pf_s = "inf" if pf == float("inf") else (f"{pf:.2f}" if pf is not None else "n/a")
    return (f"n={st.get('closed', st.get('n', 0))} WR={st.get('wr')} "
            f"avgR={st.get('avg_r')} sumR={st.get('sum_r')} PF={pf_s}")


def _render(r: dict) -> str:
    q = r["q"]
    oos = r["oos"]
    return "\n".join([
        "# Verdict: Selection Validator (quality-band)",
        "",
        f"Generated: {r['generated']}  |  OOS boundary: {r['boundary']}  |  "
        f"dataset rows: {r['dataset_rows']}",
        "",
        f"## Decision for quality >= {q}: **{r['decision']}**",
        "",
        f"- {r['decision_info']['why']}",
        "",
        "## OOS slice numbers (the only numbers that decide)",
        "",
        "| selector | closed | WR | avgR | sumR | PF |",
        "|---|---|---|---|---|---|",
        f"| baseline | {oos['baseline'].get('closed')} | {oos['baseline'].get('wr')} | "
        f"{oos['baseline'].get('avg_r')} | {oos['baseline'].get('sum_r')} | "
        f"{oos['baseline'].get('pf')} |",
        f"| quality>={q} | {oos['candidate'].get('closed')} | {oos['candidate'].get('wr')} | "
        f"{oos['candidate'].get('avg_r')} | {oos['candidate'].get('sum_r')} | "
        f"{oos['candidate'].get('pf')} |",
        "",
        f"- P(ΔavgR > 0) = {oos['p_delta_gt0']}  (bootstrap {harness.DRAWS} draws, "
        f"seed {harness.SEED}, one-sided; pre-registered bar: > {1 - harness.ALPHA})",
        "",
        "## Train slice (calibration only — no decision power)",
        "",
        f"- baseline: {_fmt(r['train']['baseline'])}",
        f"- candidate: {_fmt(r['train']['candidate'])}",
        "",
        "Pre-registered rule (spec.md SC-003): GO iff n>=150 OOS, "
        "P(ΔavgR>0) > 0.95 AND PF_cand >= PF_base. KILL decisive when evidence "
        "is against; INCONCLUSIVE waits for more trades.",
        "",
    ]) + "\n"


def check_dataset() -> list:
    """Leak-check the assembled dataset (must be clean before any verdict)."""
    df = load_rows(*glob.glob(DATASET_GLOB))
    return leak_check(df)


if __name__ == "__main__":
    import sys

    # Dedup first (idempotent): duplicate rows can bias stats if two replay
    # processes ever appended to one file (ES 2026-08-30). Cheap, always safe.
    from selection_validator.dedup_signals import main as dedup_main
    dedup_main()

    df = load_rows(*sys.argv[1:]) if sys.argv[1:] else load_rows(*glob.glob(DATASET_GLOB))
    problems = leak_check(df)
    print(f"loaded {len(df)} rows; leak check: {'CLEAN' if not problems else problems}")
    if problems:
        raise SystemExit("dataset not clean — rebuild before verdict (constitution IV)")

    # T008 baseline sanity: the harness must reproduce the doctrine edge on
    # the pre-OOS slice before any candidate is trusted.
    train, oos = harness.split_slices(df)
    base_train = evaluate(train, BaselineSelector())
    print(f"BASELINE SANITY (train, n={base_train['closed']}): "
          f"avgR={base_train['avg_r']} PF={base_train['pf']} "
          f"(doctrine: avgR 0.53-0.63 / PF 1.96-2.26)")

    for q in (5, 6, 7):
        rep = run_verdict(df, q)
        print(f"q={q}: {rep['decision']} — {rep['decision_info']['why']}")
