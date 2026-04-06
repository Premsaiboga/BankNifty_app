# brain/strategy_brain.py

# Mean-reversion strategies trade counter-trend by design,
# BUT only in RANGE regime. In TREND regime, follow the trend.
REVERSION_STRATEGIES = {"VWAP_REVERSION"}


def allow_trade(trade, bias, regime, macro_bias="NEUTRAL"):
    """
    Filter trades based on market context.

    VWAP_REVERSION:
      - RANGE regime + macro NEUTRAL → exempt (trade both directions)
      - TREND regime → must follow short-term bias (no counter-trend)
      - macro_bias SELL → block BUY (don't catch falling knives)
      - macro_bias BUY → block SELL (don't sell into rallies)

    Other strategies: blocked if short-term bias is opposite.
    """

    trade_type = trade["type"]
    strategy = trade.get("strategy", "")

    # --- Reversion strategies: regime-dependent ---
    if strategy in REVERSION_STRATEGIES:

        # Layer 1: Macro bias (multi-day trend) — strongest override
        if macro_bias == "SELL" and trade_type == "BUY":
            return False
        if macro_bias == "BUY" and trade_type == "SELL":
            return False

        # Layer 2: TREND regime — follow the short-term bias
        # In a trending market, pullbacks are NOT reversions
        if regime == "TREND":
            if bias == "BUY" and trade_type == "SELL":
                return False
            if bias == "SELL" and trade_type == "BUY":
                return False

        # RANGE/CHOPPY regime + macro NEUTRAL → allow both directions
        return True

    # --- Other strategies: use short-term bias ---
    if bias == "BUY" and trade_type == "SELL":
        return False
    if bias == "SELL" and trade_type == "BUY":
        return False

    # NEUTRAL bias → allow everything
    return True
