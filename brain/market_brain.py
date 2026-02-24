# brain/market_brain.py

import pandas as pd
import numpy as np


# ---------------- EMA ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# ---------------- BIAS DETECTION ----------------
def detect_bias(df_15m):
    """
    Returns:
        bias: BUY / SELL / NEUTRAL
        score: integer
    """

    # ---------- SAFETY CHECK ----------
    # Need minimum candles for EMA50 + structure check
    if df_15m is None or len(df_15m) < 50:
        return "NEUTRAL", 0

    required_cols = {"close", "volume"}
    if not required_cols.issubset(df_15m.columns):
        return "NEUTRAL", 0

    close = df_15m["close"]
    volume = df_15m["volume"]

    # ---------- EMA ----------
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    score = 0

    # ---------- Trend check ----------
    if ema20.iloc[-1] > ema50.iloc[-1]:
        score += 1
    else:
        score -= 1

    # ---------- Structure check ----------
    # safe because len >= 50 already
    if close.iloc[-1] > close.iloc[-5]:
        score += 1
    else:
        score -= 1

    # ---------- VWAP ----------
    cumulative_vol = volume.cumsum()

    # avoid divide-by-zero
    if cumulative_vol.iloc[-1] == 0:
        return "NEUTRAL", score

    vwap = (close * volume).cumsum() / cumulative_vol

    if close.iloc[-1] > vwap.iloc[-1]:
        score += 1
    else:
        score -= 1

    # ---------- FINAL DECISION ----------
    if score >= 2:
        return "BUY", score
    elif score <= -2:
        return "SELL", score
    else:
        return "NEUTRAL", score


# ---------------- REGIME DETECTION ----------------
def detect_regime(df_15m):
    """
    Detects market regime:
    TREND or RANGE
    """

    # Need enough candles for ATR + structure
    if df_15m is None or len(df_15m) < 14:
        return "RANGE"

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df_15m.columns):
        return "RANGE"

    high = df_15m["high"]
    low = df_15m["low"]
    close = df_15m["close"]

    # ---------- ATR proxy ----------
    atr = (high - low).rolling(14).mean()

    # protect NaN startup
    if pd.isna(atr.iloc[-1]):
        return "RANGE"

    # ---------- Recent movement ----------
    lookback = min(10, len(close) - 1)
    recent_move = abs(close.iloc[-1] - close.iloc[-lookback])

    if recent_move > atr.iloc[-1] * 2:
        return "TREND"

    return "RANGE"