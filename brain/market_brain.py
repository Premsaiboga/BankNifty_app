# brain/market_brain.py

import pandas as pd
import numpy as np


# ---------------- EMA ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# ---------------- BIAS DETECTION ----------------
def detect_bias(df_5m):
    """
    Detect market bias from 5m candles (NOT 15m — we don't have enough 15m in a day).
    Uses EMA 9/21, price structure, and price vs VWAP.

    Returns:
        bias: BUY / SELL / NEUTRAL
        score: integer
    """

    if df_5m is None or len(df_5m) < 25:
        return "NEUTRAL", 0

    required_cols = {"close", "high", "low"}
    if not required_cols.issubset(df_5m.columns):
        return "NEUTRAL", 0

    close = df_5m["close"]

    # ---------- EMA 9/21 on 5m ----------
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)

    score = 0

    # EMA trend
    if ema9.iloc[-1] > ema21.iloc[-1]:
        score += 1
    else:
        score -= 1

    # Price structure: last close vs 10 candles ago (50 mins)
    lookback = min(10, len(close) - 1)
    if close.iloc[-1] > close.iloc[-lookback]:
        score += 1
    else:
        score -= 1

    # Price vs VWAP (if available)
    if "vwap" in df_5m.columns:
        vwap_val = df_5m["vwap"].iloc[-1]
        if not pd.isna(vwap_val) and vwap_val > 0:
            if close.iloc[-1] > vwap_val:
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
def detect_regime(df_5m):
    """
    Detect market regime from 5m candles.
    TREND = directional move > 1.5x ATR over recent candles.
    RANGE = consolidating.
    """

    if df_5m is None or len(df_5m) < 14:
        return "RANGE"

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df_5m.columns):
        return "RANGE"

    high = df_5m["high"]
    low = df_5m["low"]
    close = df_5m["close"]

    # ATR on 5m candles
    atr = (high - low).rolling(14).mean()

    if pd.isna(atr.iloc[-1]):
        return "RANGE"

    # Recent directional move over 10 candles (50 mins)
    lookback = min(10, len(close) - 1)
    recent_move = abs(close.iloc[-1] - close.iloc[-lookback])

    if recent_move > atr.iloc[-1] * 1.5:
        return "TREND"

    return "RANGE"