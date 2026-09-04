#!/usr/bin/env python3
"""topx_watch.py — watches TopstepX login state; alerts on CHANGE (esp. the
moment auth starts working). Silent otherwise. Never prints the API key.

State file: ~/.topstepx_auth_state  (last result, e.g. "3" or "OK")
Alert: Telegram via local telegram.py (token from bundle .env).
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

STATE_FILE = os.path.join(os.path.expanduser("~"), ".topstepx_auth_state")


def load_username():
    for line in open(os.path.join(BASE, ".env")):
        if line.startswith("TOPSTEPX_USERNAME="):
            return line.split("=", 1)[1].strip()
    return ""


def load_key():
    try:
        with open(os.path.join(os.path.expanduser("~"), ".topstepx_api_key")) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def try_login(username, key):
    try:
        r = requests.post(
            "https://api.topstepx.com/api/Auth/loginKey",
            json={"userName": username, "apiKey": key},
            timeout=25,
        )
        d = r.json()
        return d.get("errorCode"), bool(d.get("token")), d.get("errorMessage")
    except Exception as e:
        return f"NET:{type(e).__name__}", False, str(e)[:120]


def send_tg(text):
    try:
        from telegram import send  # topstep-bot's sender
        send(text, silent=False)
    except Exception as e:
        print(f"tg alert failed: {e}", flush=True)


def main():
    username = load_username()
    key = load_key()
    if not username or not key:
        print("WATCH: missing username/key — nothing to test")
        return

    ec, has_token, msg = try_login(username, key)
    result = "OK" if has_token else str(ec)

    prev = None
    if os.path.exists(STATE_FILE):
        prev = open(STATE_FILE).read().strip()

    if prev == result:
        print(f"WATCH: no change (state={result})")
        return

    # state changed (or first run) — write state, alert only if not first run
    with open(STATE_FILE, "w") as f:
        f.write(result)

    if prev is None:
        print(f"WATCH: baseline state={result}")
        return

    if has_token:
        text = (f"\U0001F7E2 TOPSTEPX CONNECTED! Login works now.\n"
                f"username: {username}\n(previous state: {prev})\n"
                f"Next: run connect_test.py to list accounts and go live.")
    else:
        text = (f"\U0001F534 TOPSTEPX auth state changed: errorCode {prev} -> {ec}\n"
                f"msg: {msg}")
    print(f"WATCH: state changed {prev} -> {result} — alerting")
    send_tg(text)


if __name__ == "__main__":
    main()
