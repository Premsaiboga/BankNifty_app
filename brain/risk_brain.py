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
        trade["sl"] = entry - current_atr
    else:
        trade["target"] = entry - max_move
        trade["sl"] = entry + current_atr

    return trade


def reversal_warning(df_5m):
    """
    Simple exhaustion detection
    """

    last = df_5m["close"].iloc[-1]
    prev = df_5m["close"].iloc[-5]

    momentum = last - prev

    if abs(momentum) < 10:
        return True

    return False