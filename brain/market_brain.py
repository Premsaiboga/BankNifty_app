# brain/market_brain.py
import pandas as pd
import numpy as np

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def detect_bias(df_15m):
    """
    Returns:
        bias: BUY / SELL / NEUTRAL
        score: integer
    """

    close = df_15m["close"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    score = 0

    # Trend check
    if ema20.iloc[-1] > ema50.iloc[-1]:
        score += 1
    else:
        score -= 1

    # Structure check
    if close.iloc[-1] > close.iloc[-5]:
        score += 1
    else:
        score -= 1

    # VWAP position (simple proxy)
    vwap = (df_15m["close"] * df_15m["volume"]).cumsum() / df_15m["volume"].cumsum()

    if close.iloc[-1] > vwap.iloc[-1]:
        score += 1
    else:
        score -= 1

    if score >= 2:
        return "BUY", score
    elif score <= -2:
        return "SELL", score
    else:
        return "NEUTRAL", score


def detect_regime(df_15m):
    """Trend or Range detection"""

    atr = (df_15m["high"] - df_15m["low"]).rolling(14).mean()

    recent_move = abs(df_15m["close"].iloc[-1] - df_15m["close"].iloc[-10])

    if recent_move > atr.iloc[-1] * 2:
        return "TREND"
    return "RANGE"