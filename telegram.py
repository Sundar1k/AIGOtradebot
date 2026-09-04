"""Telegram alert sender via user's bot token."""
import requests

import config


def send(text, silent=True):
    # Defensive: the token loader was removed from config (TG alerts disabled
    # 2026-08-21). Never let an alert path crash the caller (watchdog).
    try:
        loader = getattr(config, "load_telegram_token", None)
        token = loader() if callable(loader) else None
        chat = getattr(config, "TELEGRAM_CHAT", None)
    except Exception as e:
        print(f"  (telegram token unavailable — skipped alert: {e})", flush=True)
        return False
    if not token or not chat:
        print("  (no telegram token — skipped alert)", flush=True)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat,
            "text": text,
            "disable_notification": silent,
        }, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"  telegram error: {e}", flush=True)
        return False
