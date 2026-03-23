# brain/market_brain.py

import pandas as pd
import numpy as np


# ---------------- EMA ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


# ---------------- BIAS DETECTION ----------------
def detect_bias(df_5m):
    """
    Detect market bias from 5m candles.
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
    TREND   = strong directional move (> 2x ATR over 10 candles)
    CHOPPY  = many direction changes, price going nowhere (sideways chop)
    RANGE   = everything else (calm, low movement)
    """

    if df_5m is None or len(df_5m) < 14:
        return "RANGE"

    required_cols = {"high", "low", "close", "open"}
    if not required_cols.issubset(df_5m.columns):
        return "RANGE"

    high = df_5m["high"]
    low = df_5m["low"]
    close = df_5m["close"]
    opn = df_5m["open"]

    # ATR on 5m candles
    atr = (high - low).rolling(14).mean()

    if pd.isna(atr.iloc[-1]) or atr.iloc[-1] < 1:
        return "RANGE"

    atr_val = atr.iloc[-1]

    # Recent directional move over 10 candles (50 mins)
    lookback = min(10, len(close) - 1)
    recent_move = abs(close.iloc[-1] - close.iloc[-lookback])

    # --- TREND: Strong directional move ---
    if recent_move > atr_val * 2.0:
        return "TREND"

    # --- CHOPPY: Many direction reversals in recent candles ---
    # Count how many times candle direction flips (green→red or red→green)
    n_check = min(10, len(close))
    recent = df_5m.iloc[-n_check:]
    directions = (recent["close"] > recent["open"]).astype(int)
    flips = (directions.diff().abs().sum())  # number of direction changes

    # Also check: price range vs sum of candle ranges (efficiency ratio)
    # Low efficiency = lots of movement but going nowhere = choppy
    total_candle_range = (recent["high"] - recent["low"]).sum()
    net_move = abs(recent["close"].iloc[-1] - recent["close"].iloc[0])
    efficiency = net_move / total_candle_range if total_candle_range > 0 else 0

    # CHOPPY = many flips (>= 5 out of 10) AND low efficiency (< 0.25)
    if flips >= 5 and efficiency < 0.25:
        return "CHOPPY"

    # Also CHOPPY if very low efficiency even with fewer flips
    if efficiency < 0.15 and n_check >= 8:
        return "CHOPPY"

    return "RANGE"
