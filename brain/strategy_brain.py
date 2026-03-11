# brain/strategy_brain.py

# Mean-reversion strategies are DESIGNED to trade counter-trend
REVERSION_STRATEGIES = {"VWAP_REVERSION"}

def allow_trade(trade, bias, regime):
    """
    Light filter — only block clear counter-trend trades for TREND strategies.
    VWAP_REVERSION is exempt (it's supposed to be counter-trend).
    NEUTRAL bias allows ALL trades.
    """

    trade_type = trade["type"]
    strategy = trade.get("strategy", "")

    # Reversion strategies are allowed to trade counter-trend
    if strategy in REVERSION_STRATEGIES:
        return True

    # Only block if bias is clearly opposite to trade direction
    if bias == "BUY" and trade_type == "SELL":
        return False

    if bias == "SELL" and trade_type == "BUY":
        return False

    # NEUTRAL bias → allow everything
    return True
