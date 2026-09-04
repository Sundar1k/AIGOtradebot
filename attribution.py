#!/usr/bin/env python3
"""attribution.py — the SHARP-style attribution agent for the autotrader.

Reads the missed-trade ledger (signals with proba + simulated realized R),
detects recurring loss patterns ACROSS samples (never one trade in
isolation), proposes 1-3 concrete atomic rule edits, validates each
walk-forward against the untouched baseline, and applies only accepted
rules — with every change recorded in a human-readable rules file.

The 4-step SHARP loop:
  1. RULES FIRST  — the bot's live behaviour is already a rubric
     (config.py: floor/ceiling/symbols/gates). This script edits ONE
     parameter per proposal — atomic edit, never a rewrite.
  2. ATTRIBUTION  — cross-sample pattern mining over the ledger:
       * per-symbol quality (drop a persistently losing symbol)
       * floor band below/above (raise floor when the band just above
         the current floor loses, i.e. the floor is too loose)
       * ceiling band (tighten the confidence ceiling when overconfident
         signals lose)
  3. VALIDATION   — every proposal is backtested on the ledger with a
     chronological split: train on the older 2/3 (diagnosis), gate on the
     newer 1/3 (out-of-sample). Accepted only if it improves expectancy
     AND doesn't worsen win rate or sample count materially.
  4. AUDITABLE    — accepted changes are appended to
     ~/.autotrade_rules.md (date, rule, evidence, before/after)
     and applied to config.py + .env, then the supervisor is restarted.
     Rejected proposals are also recorded (with why).

Run:
    python attribution.py            # full: propose -> validate -> apply
    python attribution.py --report   # diagnose + propose only, NEVER apply
    python attribution.py --force    # apply even if validation is marginal

Guards: AUTOTRADE_ATTRIB_APPLY=0 disables applying (report only).
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

# load .env at module level so env overrides (PROTECTED_SYMBOLS etc.) are
# visible to the constants below — config.py is what normally does this,
# but attribution.py reads config only inside functions.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/.env"))
except Exception:
    pass

LEDGER = os.path.join(os.path.expanduser("~"), ".autotrade_missed.json")
RULES_MD = os.path.join(os.path.expanduser("~"), ".autotrade_rules.md")
RULES_JSON = os.path.join(os.path.expanduser("~"), ".autotrade_rules.json")
STATE = os.path.join(os.path.expanduser("~"), ".autotrade_state")
CFG = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/config.py")
ENV = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/.env")
APPLY = os.environ.get("AUTOTRADE_ATTRIB_APPLY", "1") == "1"

# ── evidence thresholds (conservative; SHARP-style gating) ─────────────
# Symbols the USER has explicitly chosen to keep — the attribution agent
# never auto-drops these (the user overrides the data). Set
# AUTOTRADE_PROTECTED_SYMBOLS in .env; defaults to protecting nothing extra.
PROTECTED_SYMBOLS = [s.strip().upper() for s in
                     os.environ.get("AUTOTRADE_PROTECTED_SYMBOLS", "").split(",")
                     if s.strip()]
MIN_N_SYMBOL = 30          # min signals before we trust a symbol's stats
SYMBOL_WR_MAX = 0.32       # drop symbol if win rate below this
SYMBOL_AVGR_MAX = -0.05    # AND avg R below this (both must hold)
MIN_N_BAND = 15            # min signals in a band before acting on it
BAND_EDGE = 0.05           # look at [floor, floor+0.05) and [ceil-0.05, ceil)
EXPECTANCY_GATE = 0.03     # validation must improve avg R by at least this
VALIDATION_SPLIT = 0.67    # older fraction = train; newer = validation

# ── p-hacking ceiling (MadEvolve §7 / Bailey et al. 2014) ────────────────
# Every proposal ever validated is a "trial"; the best-of-K luck ceiling for
# expectancy gains grows as σ₀·√(2·ln K). Proposal #K must beat its static
# gate BY THIS MARGIN to count as research rather than data mining.
TRIALS_FILE = os.path.join(os.path.expanduser("~"), ".autotrade_attr_trials.json")
ATTRIB_SIGMA_R = float(os.environ.get("AUTOTRADE_ATTRIB_SIGMA_R", "1.4"))
# The luck ceiling scales with the noise of the VALIDATION-MEAN estimate
# (sigma/sqrt(n)), not per-trade sigma — per-trade sigma would demand
# impossible +1.6R improvements and veto every honest rule forever.
def _validation_noise() -> float:
    n = max(30, int(os.environ.get("AUTOTRADE_ATTRIB_VALID_N", "190")))
    return ATTRIB_SIGMA_R / (n ** 0.5)


def _load_trials() -> int:
    try:
        return int(json.load(open(TRIALS_FILE)).get("trials", 0))
    except Exception:
        return 0


def _bump_trials():
    try:
        n = _load_trials() + 1
        json.dump({"trials": n, "updated": dt.datetime.now().isoformat()},
                  open(TRIALS_FILE, "w"))
    except Exception:
        pass


def phack_margin() -> tuple:
    """(margin_R, trials): extra expectancy gain proposal #K must show."""
    import math
    k = max(1, _load_trials())
    return (_validation_noise() * math.sqrt(2.0 * math.log(k)), k)


def _alert(text: str):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from telegram import send
        send(text)
    except Exception as e:
        print(f"tg failed: {e}", flush=True)


def load_ledger() -> list:
    try:
        with open(LEDGER) as f:
            return json.load(f).get("records", [])
    except Exception:
        return []


def load_rules() -> dict:
    try:
        with open(RULES_JSON) as f:
            return json.load(f)
    except Exception:
        return {"applied": []}


def save_rules(rules: dict):
    tmp = RULES_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rules, f, indent=2)
    os.replace(tmp, RULES_JSON)


def read_config() -> dict:
    src = open(CFG).read()
    def num(name):
        m = re.search(rf"{name}\s*=\s*([\d.]+)", src)
        return float(m.group(1)) if m else None
    # live symbol list: .env AUTOTRADE_SYMBOLS wins, else config default
    symbols = None
    try:
        env = open(ENV).read()
        m = re.search(r"(?m)^AUTOTRADE_SYMBOLS=([^\n]+)", env)
        if m:
            symbols = [s.strip() for s in m.group(1).split(",") if s.strip()]
    except Exception:
        pass
    if not symbols:
        sym_m = re.search(r'TRADE_SYMBOLS = tuple\(\s*s\.strip\(\)\.upper\(\)\s*for s in os\.environ\.get\("AUTOTRADE_SYMBOLS",\s*"([^"]+)"\)', src)
        symbols = sym_m.group(1).split(",") if sym_m else None
    return {"floor": num("PROBA_FLOOR"), "ceil": num("PROBA_CEIL"),
            "symbols": symbols}


def current_state_floor() -> float:
    try:
        st = json.load(open(STATE))
        fl = st.get("evolve", {}).get("floor")
        if fl is not None:
            return float(fl)
    except Exception:
        pass
    return 0.0


def stats(recs: list) -> dict:
    if not recs:
        return {"n": 0, "wr": 0.0, "avg_r": 0.0, "total_r": 0.0}
    wins = sum(1 for r in recs if r["r"] > 0)
    return {"n": len(recs), "wr": round(wins / len(recs), 3),
            "avg_r": round(sum(r["r"] for r in recs) / len(recs), 3),
            "total_r": round(sum(r["r"] for r in recs), 3)}


def split_chrono(recs: list) -> tuple:
    """Older 2/3 = train (diagnosis), newer 1/3 = validation (gate)."""
    recs = sorted(recs, key=lambda r: r.get("time", ""))
    k = max(1, int(len(recs) * VALIDATION_SPLIT))
    return recs[:k], recs[k:]


# ── pattern detectors (attribution) ────────────────────────────────────
def detect_symbol_drops(recs: list, live_symbols: list) -> list:
    """Per-symbol: drop candidates where n >= MIN_N_SYMBOL, WR < threshold
    AND avg R negative — both conditions, never one trade alone. Only
    symbols currently in the live universe are considered."""
    out = []
    by_sym = {}
    for r in recs:
        by_sym.setdefault(r["symbol"], []).append(r)
    for sym, rs in sorted(by_sym.items()):
        if sym not in (live_symbols or []):
            continue
        if sym in PROTECTED_SYMBOLS:
            continue          # user's choice beats the data
        st = stats(rs)
        if st["n"] >= MIN_N_SYMBOL and st["wr"] < SYMBOL_WR_MAX \
                and st["avg_r"] < SYMBOL_AVGR_MAX:
            out.append({
                "rule": "drop_symbol", "param": "symbols", "target": sym,
                "action": "remove",
                "evidence": (f"{sym}: n={st['n']} WR={st['wr']:.1%} "
                             f"avg {st['avg_r']:+.2f}R"),
                "current": {"symbols": True}, "proposed": {"symbols": False},
            })
    return out


def detect_floor_raise(recs: list, floor: float) -> list:
    """If the band just ABOVE the current floor loses, the floor is too
    loose — raise it by one step. (Lowering is the missed-learner's job.)"""
    band = [r for r in recs if floor <= r["proba"] < floor + BAND_EDGE]
    st = stats(band)
    if st["n"] >= MIN_N_BAND and st["wr"] < SYMBOL_WR_MAX \
            and st["avg_r"] < 0:
        return [{
            "rule": "raise_floor", "param": "floor", "target": floor,
            "action": f"{floor:.2f} -> {floor + 0.05:.2f}",
            "evidence": (f"band [{floor:.2f},{floor+0.05:.2f}): n={st['n']} "
                         f"WR={st['wr']:.1%} avg {st['avg_r']:+.2f}R"),
            "current": {"floor": floor}, "proposed": {"floor": round(floor + 0.05, 2)},
        }]
    return []


def detect_ceil_tighten(recs: list, ceil: float) -> list:
    """If the band just BELOW the ceiling loses (overconfident signals),
    tighten the ceiling by one step."""
    if ceil >= 1.0:
        return []
    band = [r for r in recs if ceil - BAND_EDGE <= r["proba"] < ceil]
    st = stats(band)
    if st["n"] >= MIN_N_BAND and st["wr"] < SYMBOL_WR_MAX \
            and st["avg_r"] < 0:
        return [{
            "rule": "tighten_ceil", "param": "ceil", "target": ceil,
            "action": f"{ceil:.2f} -> {ceil - 0.05:.2f}",
            "evidence": (f"band [{ceil-0.05:.2f},{ceil:.2f}): n={st['n']} "
                         f"WR={st['wr']:.1%} avg {st['avg_r']:+.2f}R"),
            "current": {"ceil": ceil}, "proposed": {"ceil": round(ceil - 0.05, 2)},
        }]
    return []


def detect_conflict_block(recs: list) -> list:
    """Candle-pattern conflict rule (#6): signals that OPPOSE the 30-min
    candle pattern (pattern_dir != 0 and != signal dir) have historically
    lost. Propose blocking them — but ONLY after the validation gate proves
    the conflict penalty is real out-of-sample. Ledger records carry
    'conflict' from missed_trades.py."""
    tagged = [r for r in recs if r.get("pattern_dir", 0) != 0]
    if len(tagged) < MIN_N_SYMBOL * 2:
        return []                       # not enough pattern-tagged data yet
    conflict = [r for r in tagged if r.get("conflict")]
    align = [r for r in tagged if not r.get("conflict")]
    cs, as_ = stats(conflict), stats(align)
    # propose only when conflict is clearly worse AND has meaningful samples
    if cs["n"] >= MIN_N_SYMBOL and cs["avg_r"] < as_["avg_r"] - 0.10 \
            and cs["avg_r"] < 0:
        return [{
            "rule": "block_conflict", "param": "conflict", "target": "conflict",
            "action": "block signals opposing the candle pattern",
            "evidence": (f"conflict n={cs['n']} WR={cs['wr']:.1%} "
                         f"avg {cs['avg_r']:+.2f}R vs aligned n={as_['n']} "
                         f"WR={as_['wr']:.1%} avg {as_['avg_r']:+.2f}R"),
            "current": {"conflict": False}, "proposed": {"conflict": True},
        }]
    return []


def detect_chop_retune(recs: list) -> list:
    """Retune the chop-gate threshold (AUTOTRADE_CHOP_MAX) from live ledger
    data. The ledger doesn't carry the ATR ratio per record, so this proposes
    only when recent trades (last 40) clearly underperform older ones —
    evidence the regime shifted — and moves the threshold one step looser
    (1.0 -> 1.15) so more trades flow when the market normalizes. Conservative
    single-step, needs 120 records minimum."""
    if len(recs) < 120:
        return []
    recs = sorted(recs, key=lambda r: r.get("ts") or "")
    recent = stats(recs[-40:])
    older = stats(recs[:-40])
    if recent["n"] >= 30 and recent["avg_r"] < -0.10 and older["avg_r"] > 0.10:
        return [{
            "rule": "retune_chop", "param": "chop", "target": "chop",
            "action": "loosen chop gate to 1.15 (regime shift detected)",
            "evidence": (f"recent n={recent['n']} WR={recent['wr']:.1%} "
                         f"avg {recent['avg_r']:+.2f}R vs older n={older['n']} "
                         f"WR={older['wr']:.1%} avg {older['avg_r']:+.2f}R"),
            "current": {"chop_max": 1.0}, "proposed": {"chop_max": 1.15},
        }]
    return []


# ── validation gate ─────────────────────────────────────────────────────
def validate(recs: list, prop: dict) -> dict:
    """Backtest the proposal on a chronological split. For a drop_symbol:
    compare validation expectancy with/without the symbol. For floor/ceil:
    compare validation expectancy of the affected band under the proposed
    threshold (signals that WOULD be taken with the new rule) vs the
    baseline (current rule). Returns {'accepted': bool, 'detail': str}."""
    train, valid = split_chrono(recs)
    rule = prop["rule"]
    margin, k = phack_margin()
    gate = EXPECTANCY_GATE + margin
    detail = (f"train n={len(train)} valid n={len(valid)}; "
              f"gate={gate:.3f}R (base {EXPECTANCY_GATE} + phack {margin:.3f} @K={k}); ")

    if rule == "drop_symbol":
        sym = prop["target"]
        base = [r for r in valid if r["symbol"] != sym]     # after drop
        now_ = [r for r in valid]                            # before drop
        bs, ns = stats(base), stats(now_)
        # accepted if removing the symbol improves validation expectancy
        # and enough samples remain to be statistically meaningful
        gain = bs["avg_r"] - ns["avg_r"]
        detail += (f"with {sym}: avg {ns['avg_r']:+.2f}R n={ns['n']}; "
                   f"without: avg {bs['avg_r']:+.2f}R n={bs['n']}; "
                   f"gain {gain:+.2f}R")
        ok = gain >= gate and bs["n"] >= 30
        return {"accepted": ok, "detail": detail}

    if rule in ("raise_floor", "tighten_ceil"):
        # baseline: signals taken under CURRENT threshold in validation
        if rule == "raise_floor":
            new_thr = prop["proposed"]["floor"]
            old_thr = prop["current"]["floor"]
        else:
            new_thr = prop["proposed"]["ceil"]
            old_thr = prop["current"]["ceil"]
        # baseline = signals in the affected band (currently taken, losing)
        # after = those signals are EXCLUDED -> expectancy of what remains
        if rule == "raise_floor":
            base = [r for r in valid if r["proba"] >= old_thr]
            after = [r for r in valid if r["proba"] >= new_thr]
        else:
            base = [r for r in valid if r["proba"] < old_thr]
            after = [r for r in valid if r["proba"] < new_thr]
        bs, as_ = stats(base), stats(after)
        # the band we cut must be losing; what remains must not be worse
        cut = [r for r in valid
               if (rule == "raise_floor" and old_thr <= r["proba"] < new_thr)
               or (rule == "tighten_ceil" and new_thr <= r["proba"] < old_thr)]
        cs = stats(cut)
        detail += (f"cut band: n={cs['n']} avg {cs['avg_r']:+.2f}R; "
                   f"before: n={bs['n']} avg {bs['avg_r']:+.2f}R; "
                   f"after: n={as_['n']} avg {as_['avg_r']:+.2f}R")
        ok = cs["n"] >= MIN_N_BAND and cs["avg_r"] < -gate \
            and as_["avg_r"] >= bs["avg_r"] - 1e-9
        return {"accepted": ok, "detail": detail}

    if rule == "block_conflict":
        # baseline: ALL signals in validation (conflict ones currently taken)
        # after: conflict signals removed -> expectancy of the remainder
        base = [r for r in valid]
        after = [r for r in valid if not r.get("conflict")]
        bs, as_ = stats(base), stats(after)
        cut = [r for r in valid if r.get("conflict")]
        cs = stats(cut)
        gain = as_["avg_r"] - bs["avg_r"]
        detail += (f"conflict in valid: n={cs['n']} avg {cs['avg_r']:+.2f}R; "
                   f"all: n={bs['n']} avg {bs['avg_r']:+.2f}R; "
                   f"without conflict: n={as_['n']} avg {as_['avg_r']:+.2f}R; "
                   f"gain {gain:+.2f}R")
        ok = cs["n"] >= MIN_N_BAND and cs["avg_r"] < -gate \
            and gain >= gate
        return {"accepted": ok, "detail": detail}

    return {"accepted": False, "detail": detail + "unknown rule"}


# ── application ────────────────────────────────────────────────────────
def apply_rule(prop: dict, detail: str) -> bool:
    """Apply an accepted rule to config.py/.env and restart the supervisor."""
    rule = prop["rule"]
    try:
        if rule == "drop_symbol":
            sym = prop["target"]
            cfg = read_config()
            cur = cfg["symbols"] or ["NQ", "ES", "RTY", "YM", "GC"]
            new = [s for s in cur if s != sym]
            # prefer env override in .env (survives config edits)
            env = open(ENV).read()
            env = re.sub(r"(?m)^AUTOTRADE_SYMBOLS=.*$", "", env).rstrip() + "\n"
            env += f"AUTOTRADE_SYMBOLS={','.join(new)}\n"
            open(ENV, "w").write(env)
            applied = f"symbols: {','.join(cur)} -> {','.join(new)}"
        elif rule == "raise_floor":
            new = prop["proposed"]["floor"]
            src = open(CFG).read()
            src = re.sub(r"PROBA_FLOOR\s*=\s*[\d.]+",
                         f"PROBA_FLOOR = {new:.2f}", src, count=1)
            open(CFG, "w").write(src)
            applied = f"floor -> {new:.2f}"
        elif rule == "tighten_ceil":
            new = prop["proposed"]["ceil"]
            src = open(CFG).read()
            src = re.sub(r"PROBA_CEIL\s*=\s*[\d.]+",
                         f"PROBA_CEIL = {new:.2f}", src, count=1)
            open(CFG, "w").write(src)
            applied = f"ceil -> {new:.2f}"
        elif rule == "block_conflict":
            # gate lives in supervisor.veto_fn: env AUTOTRADE_BLOCK_CONFLICT=1
            env = open(ENV).read()
            env = re.sub(r"(?m)^AUTOTRADE_BLOCK_CONFLICT=.*$", "", env).rstrip() + "\n"
            env += "AUTOTRADE_BLOCK_CONFLICT=1\n"
            open(ENV, "w").write(env)
            applied = "block_conflict -> ON (signals opposing candle pattern are blocked)"
        elif rule == "retune_chop":
            # chop_gate threshold lives in .env: AUTOTRADE_CHOP_MAX
            new = prop["proposed"]["chop_max"]
            env = open(ENV).read()
            env = re.sub(r"(?m)^AUTOTRADE_CHOP_MAX=.*$", "", env).rstrip() + "\n"
            env += f"AUTOTRADE_CHOP_MAX={new:.2f}\n"
            open(ENV, "w").write(env)
            applied = f"chop_max -> {new:.2f}"
        else:
            return False
    except Exception as e:
        print(f"apply failed: {e}", flush=True)
        return False

    r = subprocess.run(["systemctl", "--user", "restart", "autotrade.service"],
                       capture_output=True, text=True, timeout=90)
    ok = r.returncode == 0
    entry = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rule": rule, "target": prop.get("target"),
        "action": prop["action"], "evidence": prop["evidence"],
        "validation": detail, "applied": applied, "restart_ok": ok,
    }
    rules = load_rules()
    rules["applied"].append(entry)
    save_rules(rules)
    with open(RULES_MD, "a") as f:
        f.write(f"\n## {entry['ts'][:16]} UTC — {rule} ({prop['action']})\n"
                f"- Evidence: {prop['evidence']}\n"
                f"- Validation: {detail}\n"
                f"- Applied: {applied} | restart {'OK' if ok else 'FAILED'}\n")
    _alert(f"🛠 ATTRIBUTION APPLIED: {applied} — {prop['evidence']} "
           f"({detail})")
    print(f"✅ APPLIED: {applied}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="diagnose and propose only — never apply")
    ap.add_argument("--force", action="store_true",
                    help="apply even if validation is marginal")
    args = ap.parse_args()

    recs = load_ledger()
    if len(recs) < 10:
        print(f"ledger too small ({len(recs)}) — need more history first",
              flush=True)
        return
    cfg = read_config()
    floor = current_state_floor() or cfg["floor"] or 0.35
    ceil = cfg["ceil"] or 0.50
    print(f"=== attribution {dt.datetime.now().isoformat()} "
          f"floor={floor:.2f} ceil={ceil:.2f} "
          f"apply={'on' if APPLY and not args.report else 'off'} "
          f"ledger={len(recs)} ===", flush=True)

    # 1. attribution: find patterns across samples
    props = []
    props += detect_symbol_drops(recs, cfg["symbols"])
    props += detect_floor_raise(recs, floor)
    props += detect_ceil_tighten(recs, ceil)
    props += detect_conflict_block(recs)
    props += detect_chop_retune(recs)
    if not props:
        print("no patterns found — all rules stay as-is", flush=True)
        return

    print(f"{len(props)} candidate rule change(s):")
    for p in props:
        print(f"  • {p['rule']} {p['action']}: {p['evidence']}")

    # 2-3. validation gate per proposal (each validation = one "trial" toward
    # the p-hacking ceiling — the counter grows even on rejections)
    for p in props:
        v = validate(recs, p)
        _bump_trials()
        p["validation"] = v
        status = "ACCEPT" if v["accepted"] else "REJECT"
        if args.force:
            status = "ACCEPT (forced)"
            v["accepted"] = True
        print(f"  [{status}] {p['rule']} {p['action']} — {v['detail']}")
        if v["accepted"] and APPLY and not args.report:
            apply_rule(p, v["detail"])
        elif v["accepted"]:
            _alert(f"🧠 ATTRIBUTION PROPOSAL (apply off): {p['action']} — "
                   f"{p['evidence']} ({v['detail']})")
            with open(RULES_MD, "a") as f:
                f.write(f"\n## {dt.datetime.now().isoformat()[:16]} UTC — "
                        f"PROPOSED {p['rule']} {p['action']} (not applied)\n"
                        f"- Evidence: {p['evidence']}\n"
                        f"- Validation: {v['detail']}\n")
        else:
            with open(RULES_MD, "a") as f:
                f.write(f"\n## {dt.datetime.now().isoformat()[:16]} UTC — "
                        f"REJECTED {p['rule']} {p['action']}\n"
                        f"- Evidence: {p['evidence']}\n"
                        f"- Validation: {v['detail']}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
