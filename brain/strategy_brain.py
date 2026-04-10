# brain/strategy_brain.py
#
# Simple rules like a veteran trader:
# - In a TREND, trade WITH the trend (all strategies including VWAP)
# - In a RANGE, let VWAP trade both directions (reversion works in ranges)
# - NEUTRAL bias = allow everything (no clear direction yet)

REVERSION_STRATEGIES = {"VWAP_REVERSION"}


def allow_trade(trade, bias, regime):
    """
    TREND regime: ALL strategies follow the bias (no counter-trend)
    RANGE regime: VWAP_REVERSION exempt (reversion works in ranges)
    NEUTRAL bias: allow everything
    """

    trade_type = trade["type"]
    strategy = trade.get("strategy", "")

    # RANGE regime: VWAP can trade both directions
    if regime != "TREND" and strategy in REVERSION_STRATEGIES:
        return True

    # Block counter-trend trades
    if bias == "BUY" and trade_type == "SELL":
        return False
    if bias == "SELL" and trade_type == "BUY":
        return False

    # NEUTRAL bias → allow everything
    return True
