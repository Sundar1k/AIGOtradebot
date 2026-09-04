"""Ensemble signal engine for futures scalping.

Voting ensemble over three independent signals, each producing -1/0/+1:
  1. RSI(14) mean-reversion  — oversold/overbought fade (bands 35/65)
  2. EMA(10)/EMA(30) momentum — trend follow
  3. Stochastic(14,3) pullback — momentum continuation in trend direction

Net score = sum of votes. Trade when |score| >= 2 (2-of-3 agreement).

v2 (2026-08-15): RSI 35/65 + EMA 10/30 + TP=1.0xSL — backtested 2y on NQ:
0.27→1.25 trades/day, WR 44.8→50.5%, PF 1.43. See backtest_sweep.py.
"""
import numpy as np
import pandas as pd


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def stochastic(df, k=14, d=3):
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    kk = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    dd = kk.rolling(d).mean()
    return kk, dd


def compute_scores(df):
    close = df["close"]
    out = df[["datetime", "close"]].copy()

    # 1. RSI mean-reversion
    r = rsi(close)
    out["rsi"] = r
    s1 = pd.Series(0, index=df.index)
    s1[r < 35] = 1      # oversold -> buy
    s1[r > 65] = -1     # overbought -> sell
    out["s_rsi"] = s1

    # 2. EMA momentum
    ema_f = close.ewm(span=10, adjust=False).mean()
    ema_s = close.ewm(span=30, adjust=False).mean()
    s2 = pd.Series(0, index=df.index)
    s2[ema_f > ema_s] = 1
    s2[ema_f < ema_s] = -1
    out["s_ema"] = s2

    # 3. Stochastic pullback (trend alignment via EMA slope + stoch flip)
    kk, dd = stochastic(df)
    slope = ema_s.diff(5)
    s3 = pd.Series(0, index=df.index)
    s3[(slope > 0) & (kk < 30) & (kk > dd)] = 1
    s3[(slope < 0) & (kk > 70) & (kk < dd)] = -1
    out["s_stoch"] = s3

    out["score"] = out["s_rsi"] + out["s_ema"] + out["s_stoch"]
    out["atr"] = atr(df, 14)
    return out


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def last_signal(df):
    """Return (score, atr, last_close) for the most recent completed bar."""
    s = compute_scores(df)
    row = s.iloc[-1]
    return float(row["score"]), float(row["atr"]), float(row["close"]), s
