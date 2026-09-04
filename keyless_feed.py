"""keyless_feed.py — zero-signup market data for the algo bot (2026-08-31).

yfinance (Yahoo Finance): pip-installed, NO API key, NO login, NO trial.
Delayed data only (Yahoo quotes are ~15-min delayed for most markets) —
fine for research, charting, screening and the selection-validator harness;
NEVER for live execution (the live bot keeps using the TopstepX broker API).

Provides a unified get_bars() interface in the bot's own bar format
(time/open/high/low/close/volume), so existing bot code can consume it
without changes. Symbols: Yahoo futures continuous (NQ=F, ES=F, GC=F, ...)
and any stock/ETF ticker (SPY, NVDA, ...).

Honest limits (documented, not hidden):
  - delayed quotes, rate-limited (Yahoo throttles ~2k requests/hour/IP)
  - unofficial API: can break without notice (it has before)
  - intraday granularity: 1m/5m/15m/30m/1h/1d... (1m limited to ~7 days back)
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:                      # not installed -> graceful fallback
    yf = None


def available() -> bool:
    return yf is not None


def get_bars(symbol: str, interval: str = "3m", period: str = "1mo",
             tz: str = "UTC") -> Optional[pd.DataFrame]:
    """Bars in the bot's format (time/open/high/low/close/volume).

    symbol:   'NQ=F' (Yahoo futures), 'ES=F', 'SPY', 'NVDA', ...
    interval: '1m','3m','5m','15m','30m','1h','1d',...
    period:   '1d','5d','1mo','3mo','6mo','1y','2y','5y','max'
    Returns None on any failure (never raises — feed is best-effort).
    """
    if yf is None:
        return None
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):      # single-ticker cleanup
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={time_col: "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True)   # bot format: UTC
        df.columns = [str(c).lower() for c in df.columns]   # Open -> open etc.
        keep = [c for c in ("time", "open", "high", "low", "close", "volume")
                if c in df.columns]
        return df[keep].dropna(subset=["close"]).reset_index(drop=True)
    except Exception:
        return None


def cache_to_csv(symbol: str, interval: str, period: str,
                 path: str) -> Optional[str]:
    """Fetch and cache bars to a CSV (the bot's data/ layout). Returns path."""
    df = get_bars(symbol, interval, period)
    if df is None:
        return None
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import sys
    for sym in (sys.argv[1] if len(sys.argv) > 1 else "NQ=F",):
        df = get_bars(sym, interval="5m", period="5d")
        if df is None:
            print(f"{sym}: fetch FAILED (network or Yahoo throttling)")
        else:
            print(f"{sym}: {len(df)} bars | {df['time'].iloc[0]} -> "
                  f"{df['time'].iloc[-1]} | last close {df['close'].iloc[-1]:.2f}")
