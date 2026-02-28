"""
Feature extraction for ML model.
Extracts 22 features from a DataFrame row (candle with indicators).
"""

# Feature list used by the ML model - ORDER MATTERS
FEATURE_COLUMNS = [
    "strategy_encoded",
    "rsi_14",
    "ema_9_dist",
    "ema_21_dist",
    "ema_cross",
    "vwap_dist",
    "bb_position",
    "bb_width",
    "body_ratio",
    "upper_wick",
    "lower_wick",
    "candle_size_norm",
    "prev_body_ratio",
    "prev_candle_size_norm",
    "consecutive_dir",
    "day_range_position",
    "minutes_from_open",
    "hour",
    "volatility_regime",
    "atr",
    "rr",
    "sl_distance_norm",
]

STRATEGY_MAP = {
    "ORB": 0,
    "EMA_SCALP": 1,
    "VWAP_REVERSION": 2,
    "MOMENTUM_SURGE": 3,
    "PIVOT_SCALP": 4,
    # Legacy strategies
    "VWAP_PULLBACK": 2,
    "PIVOT": 4,
    "ABCD": 3,
}


def extract_features(row, strategy: str, entry: float, stoploss: float, rr: float) -> dict:
    """
    Extract ML features from a DataFrame row.
    Returns a dict with all FEATURE_COLUMNS.
    """
    atr_safe = max(row.get("atr", 1.0), 0.01)
    sl_distance = abs(entry - stoploss)

    return {
        "strategy_encoded": STRATEGY_MAP.get(strategy, 0),
        "rsi_14": row.get("rsi_14", 50.0),
        "ema_9_dist": row.get("ema_9_dist", 0.0),
        "ema_21_dist": row.get("ema_21_dist", 0.0),
        "ema_cross": row.get("ema_cross", 0),
        "vwap_dist": row.get("vwap_dist", 0.0),
        "bb_position": row.get("bb_position", 0.5),
        "bb_width": row.get("bb_width", 1.0),
        "body_ratio": row.get("body_ratio", 0.5),
        "upper_wick": row.get("upper_wick", 0.0),
        "lower_wick": row.get("lower_wick", 0.0),
        "candle_size_norm": row.get("candle_size_norm", 1.0),
        "prev_body_ratio": row.get("prev_body_ratio", 0.5),
        "prev_candle_size_norm": row.get("prev_candle_size_norm", 1.0),
        "consecutive_dir": row.get("consecutive_dir", 0),
        "day_range_position": row.get("day_range_position", 0.5),
        "minutes_from_open": row.get("minutes_from_open", 60),
        "hour": row.get("hour", 10),
        "volatility_regime": row.get("volatility_regime", 1),
        "atr": atr_safe,
        "rr": rr,
        "sl_distance_norm": sl_distance / atr_safe,
    }
