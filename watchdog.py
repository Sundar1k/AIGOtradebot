#!/usr/bin/env python3
"""watchdog.py — keeps the autotrade stack alive; alerts Telegram when it can't.

SELF-HEALING (v2, 2026-08-17):
- autotrade.service dead            -> restart, verify, alert
- heartbeat stale (>15 min)         -> restart, verify, alert
- veto.service dead / :8765 down    -> restart (model reload ~3 min), alert
- halted by breaker                 -> alert only (deliberate stop; restart won't help)

Restart guard: never restart the same service more than once per 10 min
(tracked in /tmp/autotrade_watchdog_restarts.json) — prevents restart loops.

Silent (empty stdout, exit 0) when healthy — designed for a no_agent cron tick.
Alerts go out via the bot's own Telegram sender, so this works even if Hermes
has no delivery channel.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STALE_AFTER_S = 15 * 60          # 15 min without a heartbeat = trouble
STATE = os.path.join(os.path.expanduser("~"), ".autotrade_state")
RESTART_LOG = "/tmp/autotrade_watchdog_restarts.json"
COOLDOWN_S = 600                 # min seconds between restarts of the same service
AUTOTRADE_SVC = "autotrade.service"
PAPER_SVC = "paper-trade.service"          # paper book = the active mission (live acct dead)
PAPER_STATE = os.path.join(os.path.expanduser("~"), ".autotrade_paper_state.json")
VETO_SVC = "veto.service"
VETO_URL = "http://127.0.0.1:8765/health"
GPU_BUSY_MARKER = "/tmp/autotrade_gpu_busy"   # created by training wrapper; holds wrapper PID


def _svc_active(svc: str) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", svc],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "active"


def _restart(svc: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "--user", "restart", svc],
                           capture_output=True, text=True, timeout=90)
        return r.returncode == 0
    except Exception:
        return False


def _mark_restart(svc: str):
    try:
        d = json.load(open(RESTART_LOG))
    except Exception:
        d = {}
    d[svc] = time.time()
    with open(RESTART_LOG, "w") as f:
        json.dump(d, f)


def _recently_restarted(svc: str) -> bool:
    try:
        d = json.load(open(RESTART_LOG))
        return time.time() - d.get(svc, 0) < COOLDOWN_S
    except Exception:
        return False


def _veto_alive() -> bool:
    try:
        with urllib.request.urlopen(VETO_URL, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _gpu_busy() -> bool:
    """True while a training wrapper holds the GPU (veto restart would fight it).

    Marker /tmp/autotrade_gpu_busy contains the wrapper PID; only counts as
    busy while that PID is alive (stale marker = ignored).
    """
    try:
        pid = int(open(GPU_BUSY_MARKER).read().strip())
        os.kill(pid, 0)                 # raises if the PID is gone
        return True
    except Exception:
        return False


def _load_bot_env():
    """Load creds from the bot's .env into os.environ (dotenv-free)."""
    p = os.path.join(os.path.expanduser("~"), "projects/algoTraderBot/.env")
    try:
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass


def _guardian(dry_run: bool) -> str:
    """POSITION GUARDIAN (2026-08-21): when the supervisor is dead or its
    heartbeat is stale, an open position is UNMANAGED — close it via the
    broker API. This is the fix for the 2026-08-20 incident (process died
    mid-trade, position ran overnight, account crossed the kill line).
    Never prints secrets. Dry-run reports what it WOULD do."""
    _load_bot_env()
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config as botconfig
        import broker
        client = broker.make_broker()
        client.authenticate()
        acct = client.pick_account(botconfig.ACCOUNT)
        pos = client.any_open_position(acct["id"])
        if pos is None:
            return "GUARDIAN: no open position"
        cid = pos.get("contractId") or pos.get("contract_id")
        if dry_run:
            return f"GUARDIAN(dry-run): WOULD close {cid}"
        client.close_position(acct["id"], cid)
        return f"GUARDIAN: closed {cid} (position was unmanaged)"
    except Exception as e:
        return f"GUARDIAN FAILED: {type(e).__name__}: {str(e)[:120]}"


def main():
    dry_run = "--dry-run" in sys.argv
    problems = []
    if dry_run:
        # pure report mode: guardian check only, no restarts, no side effects
        print(_guardian(True), flush=True)
        return

    # ---- autotrade: dead or stale heartbeat -> guardian + restart ----
    if not _svc_active(AUTOTRADE_SVC):
        problems.append(f"service {AUTOTRADE_SVC} is not active")
        problems.append(_guardian(False))          # close any unmanaged position
        if not _recently_restarted(AUTOTRADE_SVC):
            if _restart(AUTOTRADE_SVC):
                _mark_restart(AUTOTRADE_SVC)
                problems.append("→ restarted, waiting for heartbeat")
            else:
                problems.append("→ RESTART FAILED")
    else:
        try:
            st = json.load(open(STATE))
            beat = dt.datetime.fromisoformat(st.get("last_beat", ""))
            age = (dt.datetime.now() - beat).total_seconds()
            if age > STALE_AFTER_S:
                problems.append(f"heartbeat stale ({age/60:.0f} min) — restarting")
                problems.append(_guardian(False))  # close any unmanaged position
                if not _recently_restarted(AUTOTRADE_SVC):
                    if _restart(AUTOTRADE_SVC):
                        _mark_restart(AUTOTRADE_SVC)
                        problems.append("→ restarted, waiting for heartbeat")
                    else:
                        problems.append("→ RESTART FAILED")
            if st.get("halted"):
                problems.append(f"HALTED: {st.get('reason', '?')} "
                                f"(balance ${st.get('balance', 0):,.2f})")
        except Exception as e:
            problems.append(f"state unreadable: {type(e).__name__}")

    # ---- paper-trade: dead or stale heartbeat -> restart (mission-critical) ----
    if not _svc_active(PAPER_SVC):
        problems.append(f"service {PAPER_SVC} is not active")
        if not _recently_restarted(PAPER_SVC):
            if _restart(PAPER_SVC):
                _mark_restart(PAPER_SVC)
                problems.append("→ paper restarted, waiting for heartbeat")
            else:
                problems.append("→ PAPER RESTART FAILED")
    else:
        try:
            pst = json.load(open(PAPER_STATE))
            pbeat = dt.datetime.fromisoformat(pst.get("last_beat", ""))
            now = dt.datetime.now(dt.timezone.utc)
            if pbeat.tzinfo is None:
                now = dt.datetime.now()          # naive file (like live state)
            page = (now - pbeat).total_seconds()
            if page > STALE_AFTER_S:
                problems.append(f"paper heartbeat stale ({page/60:.0f} min) — restarting")
                if not _recently_restarted(PAPER_SVC):
                    if _restart(PAPER_SVC):
                        _mark_restart(PAPER_SVC)
                        problems.append("→ paper restarted, waiting for heartbeat")
                    else:
                        problems.append("→ PAPER RESTART FAILED")
            if pst.get("halted"):
                problems.append(f"paper HALTED: {pst.get('reason', '?')} "
                                f"(paper balance ${pst.get('start_balance', 0):,.2f})")
        except Exception as e:
            problems.append(f"paper state unreadable: {type(e).__name__}")

    # ---- veto: dead or :8765 down -> restart (SKIP while GPU busy: a training
    # wrapper owns the card — restarting would OOM both jobs) ----
    if _gpu_busy():
        print("GPU busy (training wrapper alive) — veto restart deferred", flush=True)
    elif not _svc_active(VETO_SVC):
        problems.append(f"service {VETO_SVC} is not active")
        if not _recently_restarted(VETO_SVC):
            if _restart(VETO_SVC):
                _mark_restart(VETO_SVC)
                problems.append("→ veto restarted (model reload ~3 min), "
                                "verify next tick")
            else:
                problems.append("→ VETO RESTART FAILED")
    elif not _veto_alive():
        problems.append("veto HTTP :8765 down")
        if not _recently_restarted(VETO_SVC):
            if _restart(VETO_SVC):
                _mark_restart(VETO_SVC)
                problems.append("→ veto restarted (model reload ~3 min), "
                                "verify next tick")
            else:
                problems.append("→ VETO RESTART FAILED")

    if problems:
        try:
            from telegram import send
            send("🐕 AUTOTRADE WATCHDOG: " + " | ".join(problems))
        except Exception as e:
            print(f"watchdog tg failed: {e}", flush=True)
            sys.exit(1)
        print("ALERTED: " + "; ".join(problems), flush=True)
    # exit 0 either way: cron treats as a normal tick


if __name__ == "__main__":
    main()
