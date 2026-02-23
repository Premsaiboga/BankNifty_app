# brain/liquidity_engine.py

def detect_equal_highs(df, lookback=10, tolerance=10):
    highs = df["high"].iloc[-lookback:]
    return highs.max() - highs.min() <= tolerance


def detect_equal_lows(df, lookback=10, tolerance=10):
    lows = df["low"].iloc[-lookback:]
    return lows.max() - lows.min() <= tolerance


def wick_sweep(df):
    """
    Detect stop hunt candle
    """
    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]

    if upper_wick > body * 2:
        return "SELL_SWEEP"

    if lower_wick > body * 2:
        return "BUY_SWEEP"

    return None


def liquidity_block(trade, df):
    """
    Decide whether to block trade due to liquidity conditions
    """

    sweep = wick_sweep(df)

    # Equal highs → likely stop hunt above
    if trade["type"] == "BUY" and detect_equal_highs(df):
        return True

    # Equal lows → likely stop hunt below
    if trade["type"] == "SELL" and detect_equal_lows(df):
        return True

    # Wick sweep opposite direction
    if trade["type"] == "BUY" and sweep == "SELL_SWEEP":
        return True

    if trade["type"] == "SELL" and sweep == "BUY_SWEEP":
        return True

    return False