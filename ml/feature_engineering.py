import pandas as pd
import numpy as np

BIAS_MAP = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
REGIME_MAP = {"TREND": 1, "RANGE": 0}

def build_features(trade, df, bias, regime):

    last = df.iloc[-1]

    entry = trade["entry"]

    features = {}

    # ===== EXISTING =====
    features["candle_size"] = last["high"] - last["low"]
    features["atr"] = last["atr"]

    # ===== NEW MARKET FEATURES =====
    features["market_bias"] = BIAS_MAP.get(bias, 0)
    features["market_regime"] = REGIME_MAP.get(regime, 0)

    # distance to VWAP
    features["entry_vs_vwap"] = entry - last["vwap"]

    # pivot distance
    features["distance_to_pivot"] = abs(entry - last["pivot"])

    # ATR expansion
    atr_mean = df["atr"].rolling(20).mean().iloc[-1]
    if atr_mean:
        features["atr_ratio"] = last["atr"] / atr_mean
    else:
        features["atr_ratio"] = 1

    # short momentum
    if len(df) >= 5:
        features["momentum_5"] = last["close"] - df.iloc[-5]["close"]
    else:
        features["momentum_5"] = 0

    return features