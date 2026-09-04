#!/usr/bin/env python3
"""doctor.py — weekly deep-check of the autotrade stack for SILENT failures.

Catches the class of bug the watchdog can't see: services running, candles
flowing, everything "green" — yet the bot can't trade (the 2026-08-17
quality-gate blocker is the canonical example: gate defaulted to 6, the v1
veto model never emits a score, so every entry was blocked forever).

Checks:
  1. both services active, heartbeat fresh, not halted
  2. veto HTTP :8765 healthy
  3. quality gate is inert (AUTOTRADE_QUALITY_MIN=0) — the silent killer
  4. veto unit loads the GOOD adapter (output8b), and it exists
  5. regime file fresh (<7h) and gate on
  6. events calendar fresh (<13h, TTL 12h) and has high-impact events
  7. floor config sane and evolver floor within bounds
  8. last candle in bot.log is recent (<10 min) — data actually flowing
  9. no error/traceback spam in bot.log (last 24h)

Silent (empty stdout, exit 0) when healthy — designed for a no_agent cron.
Alerts go out via the bot's own Telegram sender when problems are found.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "topstep-bot"))

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(os.path.expanduser("~"), ".autotrade_state")
REGIME = os.path.join(os.path.expanduser("~"), ".autotrade_regime.json")
EVENTS = os.path.join(os.path.expanduser("~"), ".autotrade_events.json")
BOTLOG = os.path.join(BASE, "log", "bot.log")
VETO_URL = "http://127.0.0.1:8765/health"
VETO_UNIT = os.path.join(os.path.expanduser("~"), ".config/systemd/user/veto.service")

problems = []


def check(name: str, ok: bool, detail: str = ""):
    if not ok:
        problems.append(f"❌ {name}" + (f": {detail}" if detail else ""))


def _svc_active(svc: str) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", svc],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "active"


def main():
    # 1. services + heartbeat + halt
    check("autotrade.service active", _svc_active("autotrade.service"))
    check("veto.service active", _svc_active("veto.service"))
    st = {}
    try:
        st = json.load(open(STATE))
        beat = dt.datetime.fromisoformat(st.get("last_beat", ""))
        age = (dt.datetime.now() - beat).total_seconds()
        check("heartbeat fresh (<15m)", age < 15 * 60, f"{age/60:.0f} min old")
        check("not halted", not st.get("halted"), st.get("reason", ""))
    except Exception as e:
        check("state file readable", False, str(e))

    # 2. veto HTTP
    try:
        with urllib.request.urlopen(VETO_URL, timeout=5) as resp:
            check("veto HTTP :8765", resp.status == 200, f"status {resp.status}")
    except Exception as e:
        check("veto HTTP :8765", False, str(e))

    # 2b. cross-repo coupling: supervisor imports signals.py from topstep-bot
    # (supervisor.py line ~30 sys.path.append). If that file vanishes, the
    # veto path fails closed -> ALL entries silently blocked. Check it exists.
    check("signals.py present (topstep-bot coupling)",
          os.path.exists(os.path.join(os.path.expanduser("~"), "topstep-bot/signals.py")),
          os.path.join(os.path.expanduser("~"), "topstep-bot/signals.py missing — veto path will fail closed"))

    # 3. quality gate — THE silent killer (2026-08-17)
    try:
        env = open(os.path.join(BASE, ".env")).read()
        m = re.search(r"^AUTOTRADE_QUALITY_MIN=(\d+)", env, re.M)
        qmin = int(m.group(1)) if m else 6      # default 6 = blocking!
        check("quality gate inert (AUTOTRADE_QUALITY_MIN=0)", qmin == 0,
              f"AUTOTRADE_QUALITY_MIN={qmin}")
    except Exception as e:
        check("quality gate config", False, str(e))

    # 4. veto adapter — unit must load the good adapter, and it must exist
    try:
        unit = open(VETO_UNIT).read()
        m = re.search(r"VETO_ADAPTER=(\S+)", unit)
        adapter = m.group(1) if m else None
        if not adapter:
            check("veto adapter set in unit", False, "no VETO_ADAPTER in unit")
        else:
            check("veto adapter exists", os.path.isdir(adapter), adapter)
            check("veto adapter = good (output8b)", "output8b" in adapter
                  and "qual" not in adapter, adapter)
    except Exception as e:
        check("veto unit readable", False, str(e))

    # 5. regime freshness (cron every 6h -> allow 7h) and gate on
    try:
        age_h = (time.time() - os.path.getmtime(REGIME)) / 3600
        check("regime file fresh (<7h)", age_h < 7, f"{age_h:.1f}h old")
        reg = json.load(open(REGIME))
        check("regime gate on", reg.get("gate") == "on",
              f"regime={reg.get('regime')} prob={reg.get('prob')}")
    except Exception as e:
        check("regime file", False, str(e))

    # 6. events calendar (cache TTL 12h -> allow 13h) + content
    try:
        age_h = (time.time() - os.path.getmtime(EVENTS)) / 3600
        check("events cache fresh (<13h)", age_h < 13, f"{age_h:.1f}h old")
        evs = json.load(open(EVENTS))
        hi = [e for e in evs if e.get("importance", 0) >= 3]
        check("events cache has high-impact events", len(hi) > 0,
              f"{len(hi)} events")
        # Fed events must be present (fetch_fed_events swallows exceptions ->
        # [] — a dead federalreserve.gov silently empties the FOMC blackout)
        fed = [e for e in hi if "FOMC" in str(e.get("title", "")).upper()
               or "FEDERAL OPEN MARKET" in str(e.get("title", "")).upper()]
        check("events cache has FED/FOMC events", len(fed) > 0,
              f"{len(fed)} fed events — fed calendar fetch may be failing")
    except Exception as e:
        check("events cache", False, str(e))

    # 7. floor config sane; evolver floor within [config, config+0.30]
    try:
        cfg = open(os.path.join(BASE, "config.py")).read()
        m = re.search(r"PROBA_FLOOR\s*=\s*([\d.]+)", cfg)
        cfg_floor = float(m.group(1)) if m else None
        check("config floor sane (0.20-0.50)", cfg_floor is not None
              and 0.20 <= cfg_floor <= 0.50, f"PROBA_FLOOR={cfg_floor}")
        ev_floor = st.get("evolve", {}).get("floor")
        if cfg_floor is not None and ev_floor is not None:
            ok = cfg_floor - 1e-9 <= ev_floor <= cfg_floor + 0.30 + 1e-9
            check("evolver floor in bounds", ok,
                  f"config={cfg_floor} evolver={ev_floor}")
    except Exception as e:
        check("floor config", False, str(e))

    # 8. data actually flowing — last candle recent
    try:
        last = None
        with open(BOTLOG) as f:
            for ln in f:
                if "INFO candle" in ln:
                    last = ln
        if last is None:
            check("candle log present", False, "no candle lines in bot.log")
        else:
            m = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", last)
            if m:
                t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                age = (dt.datetime.now() - t).total_seconds()
                check("candles flowing (<10m)", age < 10 * 60,
                      f"last candle {age/60:.0f} min old")
    except Exception as e:
        check("bot.log readable", False, str(e))

    # 9. error spam in bot.log (last 24h)
    try:
        cutoff = (dt.datetime.now() - dt.timedelta(hours=24)).strftime("%Y-%m-%d")
        errs = []
        with open(BOTLOG) as f:
            for ln in f:
                if re.search(r"\b(ERROR|Traceback|CRITICAL)\b", ln) \
                        and ln[:10] >= cutoff:
                    errs.append(ln.strip())
        check("no error spam (24h)", len(errs) < 5, f"{len(errs)} errors")
    except Exception as e:
        check("bot.log readable", False, str(e))

    if problems:
        try:
            from telegram import send
            send("🩺 AUTOTRADE DOCTOR: " + " | ".join(problems))
        except Exception as e:
            print(f"doctor tg failed: {e}", flush=True)
            sys.exit(1)
        print("DOCTOR: " + "; ".join(problems), flush=True)
    # exit 0 either way: cron treats as a normal tick


if __name__ == "__main__":
    main()
