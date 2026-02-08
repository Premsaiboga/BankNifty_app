class PivotStrategy:
    def __init__(self, rr=4):
        self.rr = rr

    def generate_trades(self, df):
        trades = []

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            if prev["low"] <= curr["s1"] and curr["close"] > curr["s1"]:
                trades.append({
                    "strategy": "PIVOT",
                    "type": "BUY",
                    "entry": curr["close"],
                    "stoploss": curr["low"],
                    "rr": self.rr,
                    "time": curr["datetime"],
                    "features": {
                        "vwap_distance": abs(curr["close"] - curr["pivot"]),
                        "candle_size": curr["high"] - curr["low"],
                        "atr": curr["atr"],
                        "pattern_strength": 0.0
                    }
                })

            elif prev["high"] >= curr["r1"] and curr["close"] < curr["r1"]:
                trades.append({
                    "strategy": "PIVOT",
                    "type": "SELL",
                    "entry": curr["close"],
                    "stoploss": curr["high"],
                    "rr": self.rr,
                    "time": curr["datetime"],
                    "features": {
                        "vwap_distance": abs(curr["close"] - curr["pivot"]),
                        "candle_size": curr["high"] - curr["low"],
                        "atr": curr["atr"],
                        "pattern_strength": 0.0
                    }
                })

        return trades
