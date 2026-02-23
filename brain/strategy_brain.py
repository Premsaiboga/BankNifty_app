# brain/strategy_brain.py

def allow_trade(trade, bias, regime):
    """
    trade = dict from strategy
    """

    trade_type = trade["type"]
    strategy = trade["strategy"]

    # Block counter trend
    if bias == "BUY" and trade_type == "SELL":
        return False

    if bias == "SELL" and trade_type == "BUY":
        return False

    # Strategy permission
    if regime == "TREND" and strategy == "PIVOT":
        return False

    if regime == "RANGE" and strategy == "VWAP":
        return False

    return True