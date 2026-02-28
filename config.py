# =========================
# V2 CONFIG - Designed for messy institutional flow
# =========================

# Strategy-specific AI thresholds (LOWERED from v1)
# v1 had 0.60-0.65 which filtered out everything in choppy markets
# v2 uses 0.45-0.48 because:
# 1. Better strategies generate higher-quality signals
# 2. More features give ML model better discrimination
# 3. Even 52-55% win rate is profitable with RR 1.5-2.0
STRATEGY_THRESHOLDS = {
    "ORB": 0.48,
    "EMA_SCALP": 0.46,
    "VWAP_REVERSION": 0.45,
    "MOMENTUM_SURGE": 0.48,
    "PIVOT_SCALP": 0.46,
    # Legacy (kept for backward compatibility)
    "PIVOT": 0.50,
    "VWAP_PULLBACK": 0.50,
    "ABCD": 0.50,
}

# Default risk-reward ratios per strategy
STRATEGY_RR = {
    "ORB": 2.0,
    "EMA_SCALP": 1.5,
    "VWAP_REVERSION": 1.5,
    "MOMENTUM_SURGE": 2.0,
    "PIVOT_SCALP": 1.5,
}

# Global default (used by legacy code)
RR = 2

# Max trades per day (across all strategies)
MAX_TRADES_PER_DAY = 5

# Capital config
CAPITAL = 10000
LOT_SIZE = 15
