import pandas as pd
import numpy as np


class VWAPPullbackStrategy:
    def __init__(self, max_trades_per_day=3, rr_list=[2, 3, 4]):
        self.max_trades_per_day = max_trades_per_day
        self.rr_list = rr_list
        self.trade_count = 0

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate session VWAP
        """
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    @staticmethod
    def is_bullish(candle):
        return candle["close"] > candle["open"]

    @staticmethod
    def is_bearish(candle):
        return candle["close"] < candle["open"]

    def generate_signals(self, df: pd.DataFrame):
        """
        Main strategy logic.
        Returns a list of trade signal dictionaries.
        """

        df = df.copy()
        df["vwap"] = self.calculate_vwap(df)

        signals = []

        for i in range(1, len(df)):

            if self.trade_count >= self.max_trades_per_day:
                break

            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            # Ignore tiny candles (doji / low conviction)
            candle_body = abs(curr["close"] - curr["open"])
            candle_range = curr["high"] - curr["low"]

            if candle_range == 0 or candle_body / candle_range < 0.3:
                continue

            # ================= BUY SETUP =================
            if (
                curr["close"] > curr["vwap"] and
                prev["low"] <= prev["vwap"] and
                self.is_bullish(curr)
            ):
                entry = curr["high"]
                sl = curr["low"]

                if entry <= sl:
                    continue

                for rr in self.rr_list:
                    target = entry + (entry - sl) * rr

                    signals.append({
                        "type": "BUY",
                        "entry": round(entry, 2),
                        "stoploss": round(sl, 2),
                        "target": round(target, 2),
                        "rr": rr,
                        "time": curr["datetime"]
                    })

                self.trade_count += 1
                continue

            # ================= SELL SETUP =================
            if (
                curr["close"] < curr["vwap"] and
                prev["high"] >= prev["vwap"] and
                self.is_bearish(curr)
            ):
                entry = curr["low"]
                sl = curr["high"]

                if sl <= entry:
                    continue

                for rr in self.rr_list:
                    target = entry - (sl - entry) * rr

                    signals.append({
                        "type": "SELL",
                        "entry": round(entry, 2),
                        "stoploss": round(sl, 2),
                        "target": round(target, 2),
                        "rr": rr,
                        "time": curr["datetime"]
                    })

                self.trade_count += 1

        return signals
