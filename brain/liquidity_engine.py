# brain/liquidity_engine.py

def wick_sweep(df):
    """
    Detect stop hunt candle — only block on very obvious sweeps.
    Wick must be > 3x body (was 2x, too sensitive).
    """
    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    if body < 1:
        return None  # Doji — not a sweep

    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]

    if upper_wick > body * 3:
        return "SELL_SWEEP"

    if lower_wick > body * 3:
        return "BUY_SWEEP"

    return None


def liquidity_block(trade, df):
    """
    Only block on clear stop hunt wick sweeps.
    Removed equal highs/lows check — too many false positives on BankNifty.
    """
    sweep = wick_sweep(df)

    # Wick sweep opposite to trade direction
    if trade["type"] == "BUY" and sweep == "SELL_SWEEP":
        return True

    if trade["type"] == "SELL" and sweep == "BUY_SWEEP":
        return True

    return False