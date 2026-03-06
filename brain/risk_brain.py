# brain/risk_brain.py
import numpy as np

def atr(df, period=14):
    return (df["high"] - df["low"]).rolling(period).mean().iloc[-1]


def adjust_targets(trade, df_5m):
    """
    Replace fixed RR with ATR based targets
    """

    current_atr = atr(df_5m)

    entry = trade["entry"]

    max_move = current_atr * 2.5

    if trade["type"] == "BUY":
        trade["target"] = entry + max_move
        trade["stoploss"] = entry - current_atr
    else:
        trade["target"] = entry - max_move
        trade["stoploss"] = entry + current_atr

    return trade


def reversal_warning(df_5m):
    """
    Exhaustion detection — only warn on truly exhausted moves.
    Old threshold (10 pts) was way too sensitive for BankNifty.
    """
    if len(df_5m) < 5:
        return False

    last = df_5m["close"].iloc[-1]
    prev = df_5m["close"].iloc[-5]
    momentum = abs(last - prev)

    # Use ATR-relative threshold: exhausted = moved < 0.1 ATR in 5 candles
    atr_val = (df_5m["high"] - df_5m["low"]).rolling(14).mean().iloc[-1]
    if atr_val > 0 and momentum < atr_val * 0.1:
        return True

    return False