#!/usr/bin/env python3
"""missed_trades.py — the bot's memory: what did it skip, and was it right?

Replays the EXACT live logic (same strategy, same grader, same 2R bracket
exit) over recent history, simulates every EMA-cross signal to its realized
R, and classifies each as TAKEN (proba >= floor) or MISSED (proba < floor).

LEARNING (the important part): it looks at the band just below the current
floor — the "almost taken" signals. If that band has enough samples and is
consistently profitable (n >= MIN_SAMPLES, win rate >= MIN_WINRATE,
avg R > 0), the floor is too strict — the bot is leaving money on the table
by being too picky. It then lowers PROBA_FLOOR by one step (never below
FLOOR_MIN, cooldown between changes) and restarts the supervisor so the new
floor goes live. The existing Evolver does the opposite (raises the floor
when taken trades lose), so the two together keep the floor honest.

The veto LLM is NOT simulated (live-only, 17s per call) — the missed stats
are pre-veto. The veto remains the second gate on live entries.

Usage:
    python missed_trades.py            # analyze + maybe learn (cron)
    python missed_trades.py --report   # analyze + report, NEVER change floor
    python missed_trades.py --days 30  # longer window

Guards:
    AUTOTRADE_LEARN=0  -> report only (never touch the floor)
    AUTOTRADE_LEARN_STEP, AUTOTRADE_LEARN_MIN (default 0.05 / 0.20)
    AUTOTRADE_LEARN_COOLDOWN_H (default 48h between floor changes)
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import config
from broker import make_broker
from strategies import make_strategies, embed_context

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(os.path.expanduser("~"), ".autotrade_missed.json")
CSV_DIR = os.path.join(HERE, "data")

# ── learning thresholds ────────────────────────────────────────────────
MIN_SAMPLES = int(os.environ.get("AUTOTRADE_LEARN_MIN_SAMPLES", "20"))
MIN_WINRATE = float(os.environ.get("AUTOTRADE_LEARN_WINRATE", "0.55"))
LEARN_STEP = float(os.environ.get("AUTOTRADE_LEARN_STEP", "0.05"))
FLOOR_MIN = float(os.environ.get("AUTOTRADE_LEARN_MIN", "0.20"))
COOLDOWN_H = float(os.environ.get("AUTOTRADE_LEARN_COOLDOWN_H", "48"))
BAND = float(os.environ.get("AUTOTRADE_LEARN_BAND", "0.10"))  # look at [floor-BAND, floor)
MAX_HOLD_BARS = int(os.environ.get("AUTOTRADE_LEARN_HOLD", "60"))  # 3h on 3-min
LEARN_ENABLED = os.environ.get("AUTOTRADE_LEARN", "1") == "1"


def _alert(text: str):
    """Telegram via the bot's own sender; silent if it fails."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from telegram import send
        send(text)
    except Exception as e:
        print(f"tg failed: {e}", flush=True)


def load_bars(symbol: str) -> pd.DataFrame:
    """Recent bars for replay: CSV tail (long history) + fresh broker bars
    (last ~6 days), merged and deduped by time, newest last."""
    df = pd.DataFrame()
    csv_path = os.path.join(CSV_DIR, f"{config.base_symbol(symbol)}_3min.csv")
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
            df = df.rename(columns={"datetime": "time"})
            cut = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=45)
            df = df[df["time"] >= cut]
    except Exception as e:
        print(f"csv load failed ({symbol}): {e}", flush=True)
    try:
        from broker import make_broker
        c = make_broker()
        c.authenticate()
        acct = c.pick_account(config.ACCOUNT)
        contract = c.get_active_contract(symbol)
        fresh = c.get_bars(contract["id"], config.TIMEFRAME_MIN, limit=5000)
        if len(fresh):
            df = pd.concat([df, fresh]).drop_duplicates(
                subset="time", keep="last").sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"broker fetch failed ({symbol}): {e}", flush=True)
    return df


def simulate(df: pd.DataFrame, i: int, sig) -> dict:
    """Simulate the fixed-2R bracket from signal bar i forward (conservative:
    stop wins if a bar touches both, matching sim_broker). Returns realized R."""
    sign = sig.direction
    entry = sig.entry
    stop = sig.stop
    target = entry + sign * config.RR * sig.risk
    for j in range(i + 1, min(i + MAX_HOLD_BARS + 1, len(df))):
        hi, lo = float(df["high"].iloc[j]), float(df["low"].iloc[j])
        hit_stop = (lo <= stop) if sign > 0 else (hi >= stop)
        hit_tp = (hi >= target) if sign > 0 else (lo <= target)
        if hit_stop and hit_tp:
            return {"r": -1.0, "kind": "stop"}          # conservative
        if hit_stop:
            return {"r": -1.0, "kind": "stop"}
        if hit_tp:
            return {"r": float(config.RR), "kind": "target"}
    # max-hold: exit at last close
    close = float(df["close"].iloc[min(i + MAX_HOLD_BARS, len(df) - 1)])
    r = sign * (close - entry) / sig.risk
    return {"r": r, "kind": "hold"}


def replay(symbol: str, days: int) -> list:
    """Replay the live logic over the last `days`, return per-signal records."""
    df = load_bars(symbol)
    if len(df) < config.CTX + 60:
        print(f"{symbol}: insufficient bars ({len(df)})", flush=True)
        return []
    strat = make_strategies()[0]
    config.SYMBOL = symbol                     # grading sees the right instrument
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    out = []
    t0 = time.time()
    for i in range(300, len(df) - 1):
        if df["time"].iloc[i] < start:
            continue
        win = df.iloc[i - 201:i + 1].reset_index(drop=True)
        try:
            sig = strat.detect(win)
        except Exception:
            continue
        if sig is None:
            continue
        emb = embed_context(win, len(win) - 1)
        proba, r_hat = strat.grade(win, sig, emb=emb)
        sim = simulate(df, i, sig)
        # candlestick patterns present on the signal's 30-min candle —
        # observation-only tag, so the ledger can validate pattern edges
        try:
            import candle_patterns
            pats = candle_patterns.pattern_at_time(df, df["time"].iloc[i])
            pdir = candle_patterns.pattern_direction(pats)
        except Exception:
            pats, pdir = [], 0
        # CONFLICT: pattern direction opposes the signal direction
        # (chart-pattern-detector rule: pattern/model disagreement is a
        # caution flag — tag it, let the ledger decide if it's costly)
        conflict = bool(pdir != 0 and pdir != sig.direction)
        out.append({
            "symbol": symbol, "time": str(df["time"].iloc[i]),
            "dir": sig.direction, "proba": round(float(proba), 4),
            "r_hat": round(float(r_hat), 3), "r": round(sim["r"], 3),
            "kind": sim["kind"], "patterns": pats, "pattern_dir": pdir,
            "conflict": conflict,
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
    print(f"{symbol}: {len(out)} signals in {time.time()-t0:.0f}s", flush=True)
    return out


def analyze(records: list, floor: float) -> dict:
    """Band stats for signals just below the floor — the 'almost taken' set."""
    band_recs = [r for r in records
                 if floor - BAND <= r["proba"] < floor]
    n = len(band_recs)
    if n == 0:
        return {"n": 0, "winrate": 0.0, "avg_r": 0.0, "total_r": 0.0}
    wins = sum(1 for r in band_recs if r["r"] > 0)
    return {"n": n, "winrate": round(wins / n, 3),
            "avg_r": round(sum(r["r"] for r in band_recs) / n, 3),
            "total_r": round(sum(r["r"] for r in band_recs), 3)}


def load_ledger() -> dict:
    try:
        with open(LEDGER) as f:
            return json.load(f)
    except Exception:
        return {"records": [], "last_change": None, "changes": []}


def save_ledger(ledger: dict):
    tmp = LEDGER + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, LEDGER)


def current_floor() -> float:
    """Effective floor: evolver state if present and sane, else config."""
    try:
        st = json.load(open(os.path.join(os.path.expanduser("~"), ".autotrade_state")))
        fl = st.get("evolve", {}).get("floor")
        if fl is not None and 0.10 <= fl <= 0.60:
            return float(fl)
    except Exception:
        pass
    return float(config.PROBA_FLOOR)


def apply_floor(new_floor: float, reason: str) -> bool:
    """Lower config.PROBA_FLOOR and restart the supervisor to make it live."""
    cfg_path = os.path.join(HERE, "config.py")
    try:
        src = open(cfg_path).read()
        import re
        new_src, n = re.subn(r"PROBA_FLOOR\s*=\s*[\d.]+",
                             f"PROBA_FLOOR = {new_floor:.2f}", src, count=1)
        if n != 1:
            print("apply_floor: PROBA_FLOOR not found in config.py", flush=True)
            return False
        open(cfg_path, "w").write(new_src)
    except Exception as e:
        print(f"apply_floor: {e}", flush=True)
        return False
    r = subprocess.run(["systemctl", "--user", "restart", "autotrade.service"],
                       capture_output=True, text=True, timeout=90)
    ok = r.returncode == 0
    _alert(f"🧠 LEARNED: floor {config.PROBA_FLOOR:.2f} -> {new_floor:.2f} — "
           f"{reason}")
    if not ok:
        _alert(f"⚠️ floor changed in config but supervisor restart FAILED: "
               f"{r.stderr[:120]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="analyze and report only — never change the floor")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    ledger = load_ledger()
    floor = current_floor()
    print(f"=== missed_trades {dt.datetime.now().isoformat()} "
          f"floor={floor:.2f} learn={'on' if LEARN_ENABLED and not args.report else 'off'} ===",
          flush=True)

    all_recs = []
    for sym in config.TRADE_SYMBOLS:
        all_recs.extend(replay(sym, args.days))

    # merge into ledger (dedupe by symbol+time, keep newest, cap 2000).
    # NEW records are prepended; EXISTING records are UPGRADED in place when
    # the replay carries fields the old copy lacks (pattern tags) — so a
    # re-tagged signal replaces its untagged twin instead of being dropped.
    by_key = {(r["symbol"], r["time"]): r for r in all_recs}
    upgraded = 0
    out = []
    for r in ledger["records"]:
        key = (r["symbol"], r["time"])
        fresh = by_key.get(key)
        if fresh is not None and len(fresh) > len(r):
            out.append(fresh)
            upgraded += 1
        else:
            out.append(r)
    new = [r for r in all_recs if (r["symbol"], r["time"])
           not in {(x["symbol"], x["time"]) for x in ledger["records"]}]
    if new or upgraded:
        ledger["records"] = (new + out)[:2000]
        ledger["updated"] = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"merge: {len(new)} new, {upgraded} upgraded", flush=True)

    # ── learning decision ─────────────────────────────────────────────
    band = analyze(all_recs, floor)
    print(f"band [{floor-BAND:.2f}, {floor:.2f}): n={band['n']} "
          f"winrate={band['winrate']:.1%} avg_r={band['avg_r']:+.2f} "
          f"total_r={band['total_r']:+.2f}", flush=True)

    # ── conflict stats: do signals OPPOSING the candle pattern lose? ──
    # observation-only report (chart-pattern-detector conflict rule).
    # If conflict signals consistently underperform, the attribution agent
    # can later gate them — the ledger decides, never the eyeball.
    try:
        with_pat = [r for r in all_recs if r.get("pattern_dir", 0) != 0]
        conflict = [r for r in all_recs if r.get("conflict")]
        align = [r for r in with_pat if not r.get("conflict")]
        def _st(rs):
            if not rs:
                return "n=0"
            w = sum(1 for r in rs if r["r"] > 0)
            return (f"n={len(rs)} WR={w/len(rs):.1%} "
                    f"avg={sum(r['r'] for r in rs)/len(rs):+.2f}R")
        print(f"conflict: {_st(conflict)} | aligned: {_st(align)}",
              flush=True)
    except Exception as e:
        print(f"conflict stats skipped: {e}", flush=True)

    changed = False
    if band["n"] >= MIN_SAMPLES and band["winrate"] >= MIN_WINRATE \
            and band["avg_r"] > 0:
        new_floor = max(floor - LEARN_STEP, FLOOR_MIN)
        if new_floor < floor - 1e-9:
            last = ledger.get("last_change")
            cooldown_ok = (not last or
                           (dt.datetime.fromisoformat(last) -
                            dt.datetime.now(dt.timezone.utc)).total_seconds()
                           <= -COOLDOWN_H * 3600)
            if cooldown_ok:
                reason = (f"missed band n={band['n']} WR={band['winrate']:.0%} "
                          f"avg {band['avg_r']:+.2f}R")
                if LEARN_ENABLED and not args.report:
                    if apply_floor(new_floor, reason):
                        ledger["last_change"] = dt.datetime.now(
                            dt.timezone.utc).isoformat()
                        ledger["changes"] = (ledger.get("changes") or [])[-50:]
                        ledger["changes"].append({
                            "ts": ledger["last_change"], "old": floor,
                            "new": new_floor, "reason": reason,
                            "band": band})
                        changed = True
                        print(f"✅ FLOOR {floor:.2f} -> {new_floor:.2f} "
                              f"({reason})", flush=True)
                else:
                    _alert(f"🧠 SUGGEST: floor {floor:.2f} -> {new_floor:.2f} — "
                           f"{reason}. Set AUTOTRADE_LEARN=1 to auto-apply.")
                    print(f"SUGGEST floor -> {new_floor:.2f} ({reason})",
                          flush=True)
            else:
                print("cooldown active — no floor change", flush=True)
        else:
            print(f"floor already at minimum {FLOOR_MIN:.2f}", flush=True)
    else:
        print(f"no change: need n>={MIN_SAMPLES} WR>={MIN_WINRATE:.0%} "
              f"avg_r>0 (got n={band['n']} WR={band['winrate']:.1%} "
              f"avg_r={band['avg_r']:+.2f})", flush=True)

    save_ledger(ledger)
    print(f"ledger: {len(ledger['records'])} records, {len(new)} new",
          flush=True)
    if changed:
        print("RESTARTED: floor is live", flush=True)


if __name__ == "__main__":
    main()
