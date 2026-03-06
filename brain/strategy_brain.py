# brain/strategy_brain.py

def allow_trade(trade, bias, regime):
    """
    Light filter — only block clear counter-trend trades.
    NEUTRAL bias allows ALL trades (both BUY and SELL).
    Regime info is logged but does NOT block trades.
    """

    trade_type = trade["type"]

    # Only block if bias is clearly opposite to trade direction
    if bias == "BUY" and trade_type == "SELL":
        return False

    if bias == "SELL" and trade_type == "BUY":
        return False

    # NEUTRAL bias → allow everything (this was the main bug)
    # Regime → no longer blocks (AI filter handles quality)
    return True