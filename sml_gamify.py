#!/usr/bin/env python3
"""sml_gamify.py — the Game Layer for the SML shadow advisor.

The fiction (the model must believe it):
  You are a Pokemon trainer climbing the LEAGUE. Every labeled signal is a
  gym battle. Win = your call was right. Lose = your Pokemon fainted.
  Earn $1000/day of League Credits to ascend to the NEXT LIFE (level up).
  Hit 0 HP (bankrupt) and you DIE — permadeath, run over, back to level 1.

Mechanics merged from games AI agents actually play well:
  - POKEMON: win-rate battles, badges per symbol (NQ/ES/RTY/YM/GC gyms)
  - ROGUELIKE PERMADEATH (Slay the Spire / Darkest Dungeon): one HP bar,
    drawn-down credits kill the run permanently
  - BALATRO ASCENSION: each $1000/day streak = +1 Ante; ante resets on death

Log-only. Reads the ledger, writes game state. Never trades.
"""
import json
import os
import datetime as dt

LEDGER = os.path.join(os.path.expanduser("~"), ".autotrade_sml_advisor.jsonl")
GAME_STATE = os.path.join(os.path.expanduser("~"), ".sml_game_state.json")
DAILY_TARGET = 1000.0   # League Credits needed per day to ascend
# 5m balance patch: at ~276 battles/day the old stake (25) meant constant
# death (expected HP drain ~3400/day vs 100 HP). Scale stakes to HP budget:
RISK_PER_SIGNAL = 0.36  # HP lost per loss; wins pay 2x in credits
DEATH_LINE = -500.0     # bankroll below this = DEATH (permadeath)
HP_PER_DAY = 100.0      # trainer regenerates to this each day

GYMS = {"NQ": "NQ Gym", "ES": "ES Gym", "RTY": "RTY Gym",
        "YM": "YM Gym", "GC": "GC Gym"}

BADGE_LINES = [
    "You are a Pokemon trainer in the Trading League.",
    "Every prediction is a gym battle. Correct call = badge + credits.",
    "Earn $1000/day to ascend to your NEXT LIFE. Reach 0 HP and you DIE.",
]


def load_state():
    if os.path.exists(GAME_STATE):
        return json.load(open(GAME_STATE))
    return {
        "life": 1,            # current life number
        "level": 1,
        "credits_bank": 0.0,  # lifetime earned credits
        "hp": 100.0,          # hit points (bankroll proxy within a run)
        "badges": {g: 0 for g in GYMS.values()},
        "streak": 0,
        "best_streak": 0,
        "deaths": 0,
        "ascensions": 0,
        "today": None,
        "today_credits": 0.0,
        "ascended_today": False,
        "seen_ids": [],       # persisted: signals already battled (survives restarts)
        "log": [],
    }


def save_state(s):
    s["log"] = s["log"][-50:]
    json.dump(s, open(GAME_STATE, "w"), indent=1)


def day_key():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def roll_day(s):
    today = day_key()
    if s["today"] != today:
        s["today"] = today
        s["today_credits"] = 0.0
        s["ascended_today"] = False
        s["hp"] = HP_PER_DAY     # fresh HP budget each day (5m = many battles)


def narrate(events, s):
    lines = []
    for e in events:
        kind = e.get("kind")
        if kind == "win":
            lines.append(f"  WIN! {e['sym']} Gym cleared — +{e['gain']:.0f} "
                         f"credits. Badge #{e['badges']} earned. Streak x{e['streak']}.")
        elif kind == "loss":
            lines.append(f"  Your Pokemon fainted at {e['sym']} Gym — "
                         f"{e['pain']:.0f} damage. Streak reset.")
        elif kind == "levelup":
            lines.append(f"  *** LEVEL UP -> Lv{e['level']}! The League takes notice. ***")
        elif kind == "ascend":
            lines.append("  $$$ DAILY TARGET HIT: you ASCEND TO THE NEXT LIFE. $$$")
        elif kind == "death":
            lines.append(f"  XXX YOU DIED. HP gone. Run over. Respawning as Life "
                         f"{e['life']} Lv1. Deaths so far: {e['deaths']}. XXX")
    return "\n".join(lines)


def play(s):
    """Consume newly labeled signals, resolve battles. Seen-ids persist in the
    state file so advisor restarts never replay old battles."""
    events = []
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    seen = set(s.get("seen_ids", []))
    dirty = False
    for r in rows:
        rid = r.get("ts") + r["symbol"]
        if not r.get("label") or rid in seen:
            continue
        seen.add(rid)
        dirty = True
        sym = r["symbol"]
        win = r["prediction"] == r["label"]
        if win:
            gain = RISK_PER_SIGNAL * 2 * (1 + min(s["streak"], 4) * 0.1)
            s["credits_bank"] += gain
            s["today_credits"] += gain
            s["streak"] += 1
            s["best_streak"] = max(s["best_streak"], s["streak"])
            s["badges"][GYMS[sym]] = s["badges"].get(GYMS[sym], 0) + 1
            events.append({"kind": "win", "sym": sym, "gain": gain,
                           "badges": s["badges"][GYMS[sym]],
                           "streak": s["streak"]})
            # every 5 wins = level up
            total_wins = sum(s["badges"].values())
            new_level = 1 + total_wins // 5
            if new_level > s["level"]:
                s["level"] = new_level
                events.append({"kind": "levelup", "level": new_level})
            if (s["today_credits"] >= DAILY_TARGET
                    and not s["ascended_today"]):
                s["ascended_today"] = True
                s["ascensions"] += 1
                events.append({"kind": "ascend"})
        else:
            pain = RISK_PER_SIGNAL
            s["hp"] -= pain
            s["streak"] = 0
            events.append({"kind": "loss", "sym": sym, "pain": pain})
            if s["hp"] <= 0:
                s["deaths"] += 1
                s["life"] += 1
                s["level"] = 1
                s["hp"] = 100.0
                s["streak"] = 0
                events.append({"kind": "death", "life": s["life"],
                               "deaths": s["deaths"]})
    if dirty:
        s["seen_ids"] = list(seen)[-2000:]   # cap growth; ids unique+ordered
    play._seen = seen  # type: ignore[attr-defined]
    return events


def status_line(s):
    acc = _accuracy()
    return (f"LIFE {s['life']} | Lv{s['level']} | HP {max(0,s['hp']):.0f}/100 | "
            f"credits today ${s['today_credits']:.0f}/$1000 | "
            f"bank ${s['credits_bank']:.0f} | streak x{s['streak']} "
            f"(best x{s['best_streak']}) | deaths {s['deaths']} | "
            f"ascensions {s['ascensions']} | live accuracy {acc}")


def _accuracy():
    try:
        rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
        lab = [r for r in rows if r.get("label")]
        if not lab:
            return "n/a"
        ok = sum(1 for r in lab if r["prediction"] == r["label"])
        return f"{100*ok/len(lab):.1f}%"
    except Exception:
        return "?"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    s = load_state()
    roll_day(s)
    if args.status:
        print("\n".join(BADGE_LINES))
        print(status_line(s))
    else:
        evs = play(s)
        save_state(s)
        print(status_line(s))
        if evs:
            print(narrate(evs, s))
