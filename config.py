# =========================
# V3 CONFIG - Pullback-entry strategies with fixed-target live exits
# =========================

# Strategy-specific AI thresholds.
# The older 0.35 threshold produced many low-quality alerts. Backtests on the
# stored BankNifty data needed a stricter floor than 0.35. These values favor
# fewer, cleaner alerts while keeping the macro trend filter active.
STRATEGY_THRESHOLDS = {
    "ORB": 0.65,
    "MOMENTUM_SURGE": 0.70,
    "PIVOT_SCALP": 0.65,
    "VWAP_REVERSION": 0.70,
    "EMA_SCALP": 0.65,
}

# Candidate risk-reward ratios per strategy. Live exits are governed by
# EXIT_TARGET_R below.
STRATEGY_RR = {
    "ORB": 2.0,
    "EMA_SCALP": 2.0,
    "VWAP_REVERSION": 2.0,
    "MOMENTUM_SURGE": 2.0,
    "PIVOT_SCALP": 2.0,
}

# Max trades per day (across all strategies). The active-trade cap and AI floors
# usually keep actual frequency much lower.
MAX_TRADES_PER_DAY = 4

# One active BankNifty position at a time. Multiple strategy names can describe
# the same underlying bet, so stacking them doubles losses without adding real
# diversification.
MAX_ACTIVE_TRADES = 1
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
DAILY_MAX_LOSS_R = 2.0
MAX_CONSECUTIVE_LOSSES = 2

# Theta time-stop. You buy OPTIONS off these index alerts, and a flat option
# bleeds premium every minute (theta). If the trade hasn't reached target within
# this many minutes, exit — you're paying decay to hold a bet that isn't working.
# This is the #1 defensible risk control for an option buyer; see option_backtest.py.
TIME_STOP_MINUTES = 30

# Capital config
CAPITAL = 10000
LOT_SIZE = 15
MAX_DAILY_LOSS = 3000
