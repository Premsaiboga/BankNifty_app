# =========================
# V3 CONFIG - Pullback-entry strategies with 1:2 RR
# =========================

# Strategy-specific AI thresholds.
# The older 0.35 threshold produced many low-quality alerts. Backtests on the
# stored BankNifty data needed a stricter floor than 0.35. These values target
# roughly 3 trades/day while keeping the macro trend filter active.
STRATEGY_THRESHOLDS = {
    "ORB": 0.50,
    "MOMENTUM_SURGE": 0.50,
    "PIVOT_SCALP": 0.50,
    "VWAP_REVERSION": 0.50,
    "EMA_SCALP": 0.50,
}

# Default risk-reward ratios per strategy (1:2 RR)
STRATEGY_RR = {
    "ORB": 2.0,
    "EMA_SCALP": 2.0,
    "VWAP_REVERSION": 2.0,
    "MOMENTUM_SURGE": 2.0,
    "PIVOT_SCALP": 2.0,
}

# Max trades per day (across all strategies). Set above the 3/day target so
# strong days are not cut off too early.
MAX_TRADES_PER_DAY = 6

# Controlled overlap for trade frequency. 1 is safest but too quiet; 3 gives
# enough room for about 3 trades/day without fully unlimited stacking.
MAX_ACTIVE_TRADES = 3
ONE_ACTIVE_TRADE = MAX_ACTIVE_TRADES == 1

# Exit plan. Backtests show the old always-trail exit gives away too many
# intraday wins. A fixed 1.2R target keeps the profitable-trade rate above 50%
# while materially improving net R.
USE_FIXED_TARGET_EXIT = True
EXIT_TARGET_R = 1.2
TRAIL_FIXED_TARGET_SL = True
TRAIL_BREAKEVEN_R = 0.8
TRAIL_LOCK_TRIGGER_R = 1.0
TRAIL_LOCK_PROFIT_R = 0.3

# Daily damage control. If the first few trades are wrong, stop for the day
# instead of trying to win it back in chop.
DAILY_MAX_LOSS_R = 3.0
MAX_CONSECUTIVE_LOSSES = 3

# Capital config
CAPITAL = 10000
LOT_SIZE = 15
MAX_DAILY_LOSS = 3000
