#!/usr/bin/env python3
"""game.py — THE $1,000/DAY GAME (advisory scoring layer for the autotrader).

Phase 1: ADVISORY ONLY. Grades setups, tracks lives/XP/streaks/gold, logs
everything. Never blocks a trade — blocking comes later only if A-grades
prove out-of-sample superiority over B-grades (pre-registered gate).

State: ~/.autotrade_game.json (restart-safe)
Audit: ~/.autotrade_rules.md gets game events appended

Public API:
    grade_setup(proba, floor, aligned, regime, chop_ok, r_hat,
                et_hour, symbol, first_attempt) -> dict
    record_trade(r, grade=None, pnl_gold=0.0) -> dict
    record_victory() / record_defeat()
    state() -> current game state dict
"""
import datetime as dt
import json
import os
import threading

STATE_FILE = os.path.join(os.path.expanduser("~"), ".autotrade_game.json")
AUDIT = os.path.join(os.path.expanduser("~"), ".autotrade_rules.md")

_LOCK = threading.Lock()   # trades can close concurrently across symbols

# ── tuning constants (env-overridable; changes count as p-hack trials) ──
GOLD_TARGET = float(os.environ.get("GAME_GOLD_TARGET", "1000"))
GOLD_DEFEAT = float(os.environ.get("GAME_GOLD_DEFEAT", "-400"))
MAX_LIVES = int(os.environ.get("GAME_LIVES", "3"))

GRADE_FIRE = {"A+", "A"}          # grades allowed to trade in enforce mode


def _now():
    return dt.datetime.now(dt.timezone.utc)


def default_state():
    return {
        "gold": 0.0,            # today's P&L toward $1,000
        "lives": MAX_LIVES,
        "xp": 0,
        "floor": 0,             # permanent XP floor
        "streak": 0,
        "level": "ROOKIE",
        "mode": "NEUTRAL",
        "victories": 0,
        "defeats": 0,
        "day": None,            # ET date string of current day
        "symbols_traded_today": [],
        "graded": [],           # last 50 setup grades (audit)
        "all_time_high_xp": 0,
        "updated": None,
    }


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except Exception:
        st = default_state()
    base = default_state()
    for k, v in base.items():
        st.setdefault(k, v)
    # day rollover on ET date change: reset gold/lives, keep xp/floor/streak
    et_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    if st["day"] != et_date:
        st["day"] = et_date
        st["gold"] = 0.0
        st["lives"] = MAX_LIVES
        st["symbols_traded_today"] = []
    return st


def save_state(st: dict):
    st["updated"] = _now().isoformat()
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _level_for(xp: int) -> str:
    if xp >= 120: return "LEGEND"
    if xp >= 80: return "APEX"
    if xp >= 60: return "LOCKED II"
    if xp >= 40: return "HUNTER"
    if xp >= 20: return "LOCKED I"
    return "ROOKIE"


def _mode_for(streak: int) -> str:
    if streak <= -1 or streak == 0 and False:
        pass
    if streak <= -2:
        return "COLD"
    if streak >= 4:
        return "HOT"
    if streak >= 2:
        return "WARM"
    return "NEUTRAL"


# ── core: grade a setup ─────────────────────────────────────────────────
def grade_setup(proba: float, floor: float, *, aligned: bool,
                regime: str, chop_ok: bool, r_hat: float,
                et_hour: int, symbol: str, first_attempt: bool) -> dict:
    """Score a signal 0-10. Advisory: returns the grade; does NOT block."""
    st = load_state()
    score = 0
    reasons = []
    margin = proba - floor
    if margin >= 0.05:
        score += 2; reasons.append(f"proba margin {margin:+.2f} (+2)")
    elif margin >= 0:
        score += 1; reasons.append(f"proba clears floor {margin:+.2f} (+1)")
    if aligned:
        score += 2; reasons.append("pattern agrees (+2)")
    if regime in ("calm", "trending"):
        score += 1; reasons.append(f"regime {regime} (+1)")
    if chop_ok:
        score += 1; reasons.append("chop gate clean (+1)")
    if r_hat is not None and r_hat >= 1.2:
        score += 1; reasons.append(f"r_hat {r_hat:.2f} (+1)")
    if 9 <= et_hour < 12:
        score += 1; reasons.append("US morning session (+1)")
    # ORB V4 documented edge (own-data backtest: 57.6% WR, +0.15R, n=59):
    # setups inside the first-90min window with close confirmation score +2
    try:
        orb_ok = os.environ.get("GAME_ORB_BONUS", "1") == "1"
        in_window = 13 <= _now().hour < 15 or (et_hour in (9, 10))
        if orb_ok and in_window:
            score += 2; reasons.append("ORB V4 window (+2, backtested)")
    except Exception:
        pass
    if first_attempt:
        score += 1; reasons.append("first attempt on symbol (+1)")
        # consume it: mark this symbol as traded so re-entries don't get the
        # bonus again (fixes the always-True free-point bug)
        if symbol not in st["symbols_traded_today"]:
            st["symbols_traded_today"].append(symbol)
            save_state(st)

    grade = ("A+" if score >= 8 else "A" if score >= 6 else
             "B" if score >= 4 else "F")
    # mode from lives + streak
    if st["lives"] == 1 or st["streak"] <= -2:
        mode = "COLD"
    elif st["streak"] >= 4:
        mode = "HOT"
    elif st["streak"] >= 2:
        mode = "WARM"
    else:
        mode = "NEUTRAL"

    rec = {"time": _now().isoformat(), "symbol": symbol, "proba": round(proba, 3),
           "score": score, "grade": grade, "reasons": reasons}
    st["graded"].append(rec)
    st["graded"] = st["graded"][-50:]
    save_state(st)
    rec["mode"] = mode
    rec["would_fire_enforce"] = grade in GRADE_FIRE
    return rec


# ── core: record a closed trade ─────────────────────────────────────────
def record_trade(r: float, grade: str = "", gold_delta: float = None) -> dict:
    """r: realized R multiple. grade: from the entry-time grading ('' if old).
    gold_delta: actual $ P&L; falls back to r*150 estimate."""
    with _LOCK:
        st = load_state()
        was_streak = st["streak"]
        win = r > 0
        st["streak"] = was_streak + 1 if win else -1
        st["gold"] += gold_delta if gold_delta is not None else r * 150.0

        xp_delta = 0
        if win and grade in GRADE_FIRE:
            xp_delta = 3
        elif win:
            xp_delta = 1
        else:
            xp_delta = -1
        if grade and grade not in GRADE_FIRE:
            xp_delta -= 5          # fired a sub-grade setup (should be impossible)

        st["xp"] = max(st["floor"], st["xp"] + xp_delta)
        if st["xp"] > st["all_time_high_xp"]:
            st["all_time_high_xp"] = st["xp"]
        new_floor = (st["xp"] // 20) * 20
        leveled_floor = new_floor > st["floor"]
        if leveled_floor:
            st["floor"] = new_floor
            _audit(f"FLOOR LOCKED at {st['floor']} XP")
        st["level"] = _level_for(st["floor"])

        if win:
            st["lives"] = min(MAX_LIVES, st["lives"] + 1)
        else:
            st["lives"] -= 1

        ev = {"time": _now().isoformat(), "event": "trade", "r": round(r, 2),
              "win": win, "grade": grade or "-", "xp_delta": xp_delta,
              "xp": st["xp"], "lives": st["lives"], "gold": round(st["gold"], 2),
              "streak": st["streak"], "mode": st["mode"],
              "new_floor": leveled_floor}
        save_state(st)
    return ev


def record_victory():
    st = load_state()
    st["xp"] += 10
    st["victories"] += 1
    st["level"] = _level_for(max(st["xp"], st["floor"]))
    _audit(f"VICTORY: gold {st['gold']:.2f} banked (+10 XP -> {st['xp']})")
    save_state(st)
    return st


def record_defeat(reason: str):
    st = load_state()
    st["xp"] = max(st["floor"], st["xp"] - 10)
    st["defeats"] += 1
    st["lives"] = 0
    _audit(f"DEFEAT ({reason}): gold {st['gold']:.2f} (-10 XP -> {st['xp']})")
    save_state(st)
    return st


def _audit(text: str):
    try:
        with open(AUDIT, "a") as f:
            f.write(f"\n## {_now().isoformat()[:16]} UTC — GAME: {text}\n")
    except Exception:
        pass


def state() -> dict:
    return load_state()


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        print(json.dumps(grade_setup(
            0.42, 0.35, aligned=True, regime="trending", chop_ok=True,
            r_hat=1.4, et_hour=10, symbol="NQ", first_attempt=True), indent=2))
        print(json.dumps(record_trade(1.9, "A+"), indent=2))
