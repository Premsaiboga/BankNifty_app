# =========================
# V2 CONFIG - Synced with ai_filter_v2.py (OOS-optimized thresholds)
# =========================

# Strategy-specific AI thresholds (based on out-of-sample testing)
STRATEGY_THRESHOLDS = {
    "ORB": 0.50,
    "EMA_SCALP": 0.52,
    "VWAP_REVERSION": 0.58,
    "MOMENTUM_SURGE": 0.56,
    "PIVOT_SCALP": 0.46,
}

# Default risk-reward ratios per strategy
STRATEGY_RR = {
    "ORB": 2.0,
    "EMA_SCALP": 1.5,
    "VWAP_REVERSION": 1.5,
    "MOMENTUM_SURGE": 2.0,
    "PIVOT_SCALP": 1.5,
}

# Max trades per day (across all strategies)
MAX_TRADES_PER_DAY = 5

# Capital config
CAPITAL = 10000
LOT_SIZE = 15
MAX_DAILY_LOSS = 3000
