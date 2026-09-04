#!/usr/bin/env python3
"""market_context.py — news/event awareness for the autotrader veto.

The veto LLM was fine-tuned on indicator states only, so it can't know about
news. This module gives the bot two layers of news awareness:

  1. EVENT BLACKOUT (hard, deterministic) — no entries inside the danger
     window around high-impact macro releases (FOMC, CPI, NFP, PCE, etc.).
     Never relies on the LLM; protects the eval regardless of model quality.
  2. CONTEXT LINE (soft) — a short "NEWS: ..." prefix added to the veto
     prompt when headlines are available, so the LLM can factor the
     narrative into its decision.

Sources (all free, no API key):
  * Fed official calendar JSON (federalreserve.gov/json/calendar.json) —
    FOMC meetings, speeches, data releases. Refreshed daily.
  * Google News RSS (news.google.com/rss/search) — symbol headlines.
  * Static overrides list for known-fixed high-impact dates (NFP, CPI) that
    the Fed feed doesn't carry.

All fetches fail-open: no network -> no context, bot trades on indicators
as before. The blackout list itself is cached locally so it still works
offline once fetched.
"""
import datetime as dt
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import requests

ET_TZ = ZoneInfo("America/New_York")

# Map any of the bot's symbols (incl. micros) to a full futures name for
# news queries. The bot calls context_line(base_symbol) — see module docstring.
def base_symbol(symbol: str) -> str:
    """Strip a micro prefix: MNQ->NQ, MES->ES (mirrors config.base_symbol)."""
    s = symbol.upper()
    return {"MNQ": "NQ", "MES": "ES", "MGC": "GC", "M2K": "RTY", "MYM": "YM"}.get(s, s)
CACHE = os.path.join(os.path.expanduser("~"), ".autotrade_events.json")
CACHE_TTL = 12 * 3600          # refresh event list every 12h
NEWS_TTL = 15 * 60             # refresh headlines every 15 min
BLACKOUT_BEFORE = 15           # minutes before event: no entries
BLACKOUT_AFTER = 15            # minutes after event: no entries
MIN_IMPORTANCE = 3             # only events >= this importance trigger blackout

# Static high-impact releases the Fed feed doesn't list (NFP, CPI, PCE, GDP…).
# Format: {"name": "...", "importance": 3} + dynamic date resolution below.
# For 2026 the BLS/BoC schedules are known through the year; we encode the
# usual monthly pattern and allow manual overrides in ~/.autotrade_events.json.
STATIC_EVENTS = [
    {"name": "Nonfarm Payrolls (NFP)", "importance": 3, "weekday": 4,
     "week": 0, "hour": 8, "minute": 30},   # 1st Friday 08:30 ET
    {"name": "CPI (m/m)", "importance": 3, "weekday": 1, "week": 1,
     "hour": 8, "minute": 30},              # ~2nd Tuesday 08:30 ET
    {"name": "FOMC Rate Decision", "importance": 3, "source": "fed"},
]

_news_cache = {"ts": 0, "text": ""}
_events_cache = {"ts": 0, "events": None}


def _month_event(year: int, month: int, weekday: int, week: int,
                 hour: int, minute: int) -> dt.datetime:
    """Nth weekday of a month (week 0 = first) at ET."""
    d = dt.date(year, month, 1)
    while d.weekday() != weekday:
        d += dt.timedelta(days=1)
    d += dt.timedelta(weeks=week)
    return dt.datetime(year, month, d.day, hour, minute, tzinfo=ET_TZ)


def _static_events(now: dt.datetime) -> list:
    """Resolve the static monthly events around now (this + next month)."""
    out = []
    for ev in STATIC_EVENTS:
        if ev.get("source") == "fed":
            continue
        for delta in (0, 1):
            m = now.month + delta
            y = now.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            out.append({"title": ev["name"], "importance": ev["importance"],
                        "time": _month_event(y, m, ev["weekday"], ev["week"],
                                             ev["hour"], ev["minute"])})
    return out


def fetch_fed_events() -> list:
    """Fed official calendar JSON -> [{title, importance, time(ET), type}]."""
    try:
        r = requests.get("https://www.federalreserve.gov/json/calendar.json",
                         timeout=20)
        r.raise_for_status()
        data = json.loads(r.content.decode("utf-8-sig"))   # feed starts with a BOM
        out = []
        for ev in data.get("events", []):
            title = ev.get("title", "")
            month = ev.get("month", "")
            days = ev.get("days", "")
            tstr = ev.get("time", "")
            etype = ev.get("type", "")
            if not month or not days:
                continue
            try:
                m = dt.datetime.strptime(month, "%Y-%m")
            except ValueError:
                continue
            # days can be a single int OR a comma list ("3, 10, 17, 24, 31")
            # for weekly-recurring releases — one event per day.
            day_list = [int(d.strip()) for d in str(days).split(",") if d.strip()]
            if not day_list:
                continue
            for day in day_list:
                try:
                    t = dt.datetime(m.year, m.month, day, tzinfo=ET_TZ)
                    hm = re.search(r"(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.)", tstr)
                    if hm:
                        h, mi, ap = int(hm.group(1)), int(hm.group(2)), hm.group(3)
                        if ap == "p.m." and h != 12:
                            h += 12
                        if ap == "a.m." and h == 12:
                            h = 0
                        t = t.replace(hour=h, minute=mi)
                    imp = 3 if any(k in title.upper() for k in
                                  ("FOMC", "FEDERAL OPEN MARKET")) else 1
                    out.append({"title": title, "importance": imp, "time": t,
                                "type": etype})
                except (ValueError, OverflowError):
                    continue
        return out
    except Exception:
        return []


def _load_overrides() -> list:
    try:
        with open(os.path.join(os.path.expanduser("~"), ".autotrade_event_overrides.json")) as f:
            raw = json.load(f)
        out = []
        for e in raw:
            t = dt.datetime.fromisoformat(e["time"]).replace(tzinfo=ET_TZ)
            out.append({"title": e["title"], "importance": int(e.get("importance", 3)),
                        "time": t})
        return out
    except Exception:
        return []


def get_events(now: dt.datetime | None = None) -> list:
    """High-impact events in the near future (now .. now+36h). Cached 12h."""
    now = now or dt.datetime.now(ET_TZ)
    if time.time() - _events_cache["ts"] > CACHE_TTL or _events_cache["events"] is None:
        evs = fetch_fed_events() + _static_events(now) + _load_overrides()
        _events_cache["events"] = evs
        _events_cache["ts"] = time.time()
        try:
            with open(CACHE, "w") as f:
                json.dump([{"title": e["title"], "importance": e["importance"],
                            "time": e["time"].isoformat(), "type": e.get("type", "")}
                           for e in evs], f)
        except Exception:
            pass
    else:
        evs = _events_cache["events"]
    return [e for e in evs
            if e["importance"] >= MIN_IMPORTANCE
            and now <= e["time"] <= now + dt.timedelta(hours=36)]


def event_blackout(now: dt.datetime | None = None) -> dict | None:
    """If we're inside a high-impact event window, return the event; else None."""
    now = now or dt.datetime.now(ET_TZ)
    for e in get_events(now):
        t = e["time"]
        if now >= t - dt.timedelta(minutes=BLACKOUT_BEFORE) and \
           now <= t + dt.timedelta(minutes=BLACKOUT_AFTER):
            return e
    return None


# Google News query terms per symbol — use full names, not tickers, to avoid
# matching ETFs/indices that share the ticker (e.g. "NQ" -> an obscure EUR ETF).
SYMBOL_QUERY = {
    "NQ": "Nasdaq 100 futures",
    "ES": "S&P 500 futures",
    "GC": "gold futures",
    "RTY": "Russell 2000 futures",
    "YM": "Dow Jones futures",
    "MNQ": "Nasdaq 100 futures",
    "MES": "S&P 500 futures",
}
_JUNK = re.compile(r"(\^|ETF|Index|latest stock news)", re.I)


def get_headlines(symbol: str) -> str:
    """Google News RSS for the symbol -> "Headline one | Headline two" (cached)."""
    global _news_cache
    if time.time() - _news_cache["ts"] < NEWS_TTL and _news_cache["text"]:
        return _news_cache["text"]
    text = ""
    try:
        q = SYMBOL_QUERY.get(base_symbol(symbol), f"{symbol} futures")
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": q, "when": "1d", "hl": "en-US"},
            timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = root.findall(".//item")[:5]
        titles = []
        for it in items:
            t = it.findtext("title")
            if not t:
                continue
            t = t.split(" - ", 1)[0].strip()
            if _JUNK.search(t) or len(t) < 12:
                continue
            titles.append(t)
        text = " | ".join(titles[:3])
    except Exception:
        pass
    _news_cache = {"ts": time.time(), "text": text}
    return text


def context_line(symbol: str) -> str:
    """A short context prefix for the veto prompt, or '' if nothing useful."""
    parts = []
    ev = event_blackout()
    if ev:
        parts.append(f"⚠️ {ev['title']} at {ev['time'].strftime('%H:%M')} ET")
    else:
        near = [e for e in get_events()
                if e["time"] <= dt.datetime.now(ET_TZ) + dt.timedelta(hours=6)]
        if near:
            e = near[0]
            parts.append(f"{e['title']} in {int((e['time'] - dt.datetime.now(ET_TZ)).total_seconds()//60)} min")
    h = get_headlines(symbol)
    if h:
        parts.append(h[:220])
    return "CONTEXT: " + " || ".join(parts) + "." if parts else ""


if __name__ == "__main__":
    now = dt.datetime.now(ET_TZ)
    print("now ET:", now.strftime("%Y-%m-%d %H:%M"))
    print("blackout:", event_blackout(now))
    print("context:", context_line("NQ"))
    print("\nUpcoming events (next 36h):")
    for e in get_events(now)[:8]:
        print(f"  {e['time'].strftime('%m-%d %H:%M')}  {e['title']} (imp {e['importance']})")
