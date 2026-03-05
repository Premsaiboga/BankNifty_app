# =========================
# V3 CONFIG - Pullback-entry strategies with 1:2 RR
# =========================

# Strategy-specific AI thresholds (model overrides at load)
# All 5 strategies re-enabled with V3 pullback entry logic
STRATEGY_THRESHOLDS = {
    "ORB": 0.45,
    "MOMENTUM_SURGE": 0.45,
    "PIVOT_SCALP": 0.45,
    "VWAP_REVERSION": 0.45,
    "EMA_SCALP": 0.45,
}

# Default risk-reward ratios per strategy (1:2 RR)
STRATEGY_RR = {
    "ORB": 2.0,
    "EMA_SCALP": 2.0,
    "VWAP_REVERSION": 2.0,
    "MOMENTUM_SURGE": 2.0,
    "PIVOT_SCALP": 2.0,
}

# Max trades per day (across all strategies)
MAX_TRADES_PER_DAY = 5

# Capital config
CAPITAL = 10000
LOT_SIZE = 15
MAX_DAILY_LOSS = 3000
