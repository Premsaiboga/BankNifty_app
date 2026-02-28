"""
BankNifty Option Premium Calculator
=====================================
Calculates strike, expiry, and premium entry/SL/target.

BankNifty Options:
- Strike interval: 100 points
- Lot size: 15 (current as of 2025-2026)
- Weekly expiry: Wednesday
"""

import math
from datetime import datetime, timedelta

STRIKE_INTERVAL = 100
LOT_SIZE = 15


def get_next_expiry() -> str:
    """Get next weekly expiry date (Wednesday). Returns like '5th Mar'."""
    today = datetime.now()
    days_ahead = 2 - today.weekday()  # Wednesday = 2
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0:
        # If it's Wednesday, expiry is today (if before 3:30 PM)
        if today.hour >= 15 and today.minute >= 30:
            days_ahead = 7

    expiry = today + timedelta(days=days_ahead)

    day = expiry.day
    if 11 <= day <= 13:
        suffix = "th"
    elif day % 10 == 1:
        suffix = "st"
    elif day % 10 == 2:
        suffix = "nd"
    elif day % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"

    return expiry.strftime(f"{day}{suffix} %b")


def get_days_to_expiry() -> int:
    """Days remaining to next Wednesday expiry."""
    today = datetime.now()
    days_ahead = 2 - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0 and today.hour >= 15:
        days_ahead = 7
    return max(days_ahead, 0)


def get_atm_strike(spot_price: float) -> int:
    return round(spot_price / STRIKE_INTERVAL) * STRIKE_INTERVAL


def get_delta(spot_price: float, strike: int) -> float:
    """Approximate delta based on moneyness."""
    moneyness = abs(spot_price - strike)
    if moneyness < 50:
        return 0.50
    elif moneyness < 150:
        return 0.32
    elif moneyness < 250:
        return 0.18
    else:
        return 0.10


def estimate_entry_premium(spot_price: float, strike: int, atr: float) -> float:
    """Estimate the option premium at entry."""
    daily_atr = atr * 3.5
    moneyness = abs(spot_price - strike)

    if moneyness < 50:
        base = daily_atr * 0.40
    elif moneyness < 150:
        base = daily_atr * 0.20
    else:
        base = daily_atr * 0.10

    # Time decay adjustment
    days = get_days_to_expiry()
    if days == 0:
        base *= 0.50
    elif days == 1:
        base *= 0.70
    elif days == 2:
        base *= 0.85

    return max(round(base, 0), 15)


def get_option_recommendation(trade: dict, atr: float) -> dict:
    """
    Get complete option recommendation with premium entry/SL/target.

    Returns dict with:
        - strike, option_type, expiry_str
        - premium_entry, premium_sl, premium_target
        - alt_strike, alt_premium_entry, alt_premium_sl, alt_premium_target
    """
    spot = trade["entry"]
    index_sl = trade["stoploss"]
    index_target = trade["target"]
    trade_type = trade["type"]

    atm = get_atm_strike(spot)
    expiry = get_next_expiry()

    # Primary: 1 OTM (cheaper premium)
    if trade_type == "BUY":
        strike = atm + STRIKE_INTERVAL
        option_type = "CE"
    else:
        strike = atm - STRIKE_INTERVAL
        option_type = "PE"

    # Alternative: ATM (higher delta, more responsive)
    alt_strike = atm
    alt_option_type = option_type

    # Delta for premium calculations
    delta = get_delta(spot, strike)
    alt_delta = get_delta(spot, alt_strike)

    # Premium entry estimates
    premium_entry = estimate_entry_premium(spot, strike, atr)
    alt_premium_entry = estimate_entry_premium(spot, alt_strike, atr)

    # Index movement
    index_move_to_target = abs(index_target - spot)
    index_move_to_sl = abs(index_sl - spot)

    # Premium SL and Target
    premium_target = round(premium_entry + index_move_to_target * delta, 0)
    premium_sl = round(premium_entry - index_move_to_sl * delta, 0)
    premium_sl = max(premium_sl, 5)  # Premium can't go below 5

    alt_premium_target = round(alt_premium_entry + index_move_to_target * alt_delta, 0)
    alt_premium_sl = round(alt_premium_entry - index_move_to_sl * alt_delta, 0)
    alt_premium_sl = max(alt_premium_sl, 5)

    return {
        "strike": strike,
        "option_type": option_type,
        "expiry": expiry,
        "premium_entry": premium_entry,
        "premium_sl": premium_sl,
        "premium_target": premium_target,
        "delta": delta,
        "lot_size": LOT_SIZE,
        "capital": round(premium_entry * LOT_SIZE, 0),
        # Alternative
        "alt_strike": alt_strike,
        "alt_option_type": alt_option_type,
        "alt_premium_entry": alt_premium_entry,
        "alt_premium_sl": alt_premium_sl,
        "alt_premium_target": alt_premium_target,
        "alt_delta": alt_delta,
        "alt_capital": round(alt_premium_entry * LOT_SIZE, 0),
    }
