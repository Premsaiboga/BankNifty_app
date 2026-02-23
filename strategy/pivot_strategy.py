# strategy/pivot_strategy.py

class PivotStrategy:
    """
    Pivot Strategy V2
    - Uses pivot zones (not exact price)
    - Detects rejection or breakout
    - Prevents fake signals
    """

    def __init__(self):
        self.zone_percent = 0.0015   # 0.15% pivot zone

    # =========================
    # HELPERS
    # =========================
    def in_zone(self, price, level):
        zone = level * self.zone_percent
        return abs(price - level) <= zone

    def candle_rejection_buy(self, candle):
        body = abs(candle["close"] - candle["open"])
        lower_wick = candle["open"] - candle["low"] if candle["close"] > candle["open"] else candle["close"] - candle["low"]
        return lower_wick > body * 1.5

    def candle_rejection_sell(self, candle):
        body = abs(candle["close"] - candle["open"])
        upper_wick = candle["high"] - candle["close"] if candle["close"] > candle["open"] else candle["high"] - candle["open"]
        return upper_wick > body * 1.5

    # =========================
    # MAIN LOGIC
    # =========================
    def generate_trades(self, df):

        trades = []

        if len(df) < 20:
            return trades

        last = df.iloc[-1]
        prev = df.iloc[-2]

        pivot = last["pivot"]
        r1 = last["r1"]
        s1 = last["s1"]

        price = last["close"]

        # ======================
        # BUY FROM S1 REJECTION
        # ======================
        if s1 and self.in_zone(price, s1):

            if self.candle_rejection_buy(last):

                entry = price
                sl = last["low"] - 10

                trades.append({
                    "strategy": "PIVOT",
                    "type": "BUY",
                    "entry": entry,
                    "stoploss": sl,
                    "reason": "S1 rejection bounce"
                })

        # ======================
        # SELL FROM R1 REJECTION
        # ======================
        if r1 and self.in_zone(price, r1):

            if self.candle_rejection_sell(last):

                entry = price
                sl = last["high"] + 10

                trades.append({
                    "strategy": "PIVOT",
                    "type": "SELL",
                    "entry": entry,
                    "stoploss": sl,
                    "reason": "R1 rejection drop"
                })

        # ======================
        # PIVOT BREAKOUT BUY
        # ======================
        if pivot:

            # breakout confirmation
            if prev["close"] < pivot and last["close"] > pivot:

                entry = price
                sl = pivot - 15

                trades.append({
                    "strategy": "PIVOT",
                    "type": "BUY",
                    "entry": entry,
                    "stoploss": sl,
                    "reason": "Pivot breakout up"
                })

            # breakdown confirmation
            if prev["close"] > pivot and last["close"] < pivot:

                entry = price
                sl = pivot + 15

                trades.append({
                    "strategy": "PIVOT",
                    "type": "SELL",
                    "entry": entry,
                    "stoploss": sl,
                    "reason": "Pivot breakout down"
                })

        return trades