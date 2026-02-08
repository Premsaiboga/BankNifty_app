class ABCDStrategy:
    def __init__(self, rr=4):
        self.rr = rr

    def generate_trades(self, df):
        trades = []

        for i in range(3, len(df)):
            a, b, c, d = df.iloc[i-3], df.iloc[i-2], df.iloc[i-1], df.iloc[i]

            if b["high"] > a["high"] and c["low"] > a["low"] and d["close"] > c["close"]:
                trades.append({
                    "strategy": "ABCD",
                    "type": "BUY",
                    "entry": d["close"],
                    "stoploss": d["low"],
                    "rr": self.rr,
                    "time": d["datetime"],
                    "features": {
                        "vwap_distance": abs(d["close"] - d["vwap"]),
                        "candle_size": d["high"] - d["low"],
                        "atr": d["atr"],
                        "pattern_strength": 1.0
                    }
                })

            elif b["low"] < a["low"] and c["high"] < a["high"] and d["close"] < c["close"]:
                trades.append({
                    "strategy": "ABCD",
                    "type": "SELL",
                    "entry": d["close"],
                    "stoploss": d["high"],
                    "rr": self.rr,
                    "time": d["datetime"],
                    "features": {
                        "vwap_distance": abs(d["close"] - d["vwap"]),
                        "candle_size": d["high"] - d["low"],
                        "atr": d["atr"],
                        "pattern_strength": 1.0
                    }
                })

        return trades
