# brain/strategy_brain.py

# Mean-reversion strategies trade counter-trend by design,
# BUT only when the macro trend is neutral (no strong multi-hour trend).
# In a strong downtrend, buying dips = catching falling knives.
REVERSION_STRATEGIES = {"VWAP_REVERSION"}


def allow_trade(trade, bias, regime, macro_bias="NEUTRAL"):
    """
    Filter trades based on market context.

    - Trend-following strategies: blocked if short-term bias is opposite
    - VWAP_REVERSION: allowed counter-trend ONLY if macro_bias is NEUTRAL
      If macro_bias is SELL → block BUY reversions (don't catch falling knives)
      If macro_bias is BUY → block SELL reversions (don't sell into rallies)
    - NEUTRAL bias → allow everything
    """

    trade_type = trade["type"]
    strategy = trade.get("strategy", "")

    # --- Reversion strategies: conditional exemption ---
    if strategy in REVERSION_STRATEGIES:
        # If no strong macro trend, allow counter-trend reversions
        if macro_bias == "NEUTRAL":
            return True

        # If strong macro trend, only allow reversions WITH the macro trend
        # (e.g., macro=SELL → allow SELL reversions, block BUY reversions)
        if macro_bias == "SELL" and trade_type == "BUY":
            return False
        if macro_bias == "BUY" and trade_type == "SELL":
            return False

        return True

    # --- Trend-following strategies: use short-term bias ---
    if bias == "BUY" and trade_type == "SELL":
        return False
    if bias == "SELL" and trade_type == "BUY":
        return False

    # NEUTRAL bias → allow everything
    return True
