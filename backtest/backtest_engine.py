import pandas as pd
import numpy as np
from strategy.vwap_strategy import VWAPPullbackStrategy


class BacktestEngine:
    def __init__(self, rr_list=[2, 3, 4]):
        self.rr_list = rr_list

    @staticmethod
    def calculate_atr(df, period=14):
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def evaluate_trade(self, df, start_index, trade):
        """
        Check candle-by-candle whether SL or Target is hit
        """
        entry = trade["entry"]
        sl = trade["stoploss"]
        target = trade["target"]
        trade_type = trade["type"]

        for i in range(start_index + 1, len(df)):
            candle = df.iloc[i]

            if trade_type == "BUY":
                if candle["low"] <= sl:
                    return "LOSS"
                if candle["high"] >= target:
                    return "WIN"

            if trade_type == "SELL":
                if candle["high"] >= sl:
                    return "LOSS"
                if candle["low"] <= target:
                    return "WIN"

        return "BE"

    def run(self, csv_path, output_path):
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])

        strategy = VWAPPullbackStrategy(max_trades_per_day=3)
        df["vwap"] = strategy.calculate_vwap(df)
        df["atr"] = self.calculate_atr(df)

        signals = strategy.generate_signals(df)
        if not signals:
            print("No trade signals generated. Check strategy conditions.")
            return
        trade_logs = []

        for trade in signals:
            entry_time = pd.to_datetime(trade["time"])
            start_index = df.index[df["datetime"] == entry_time][0]

            result = self.evaluate_trade(df, start_index, trade)

            candle = df.loc[start_index]

            trade_logs.append({
                "type": trade["type"],
                "entry": trade["entry"],
                "stoploss": trade["stoploss"],
                "target": trade["target"],
                "rr": trade["rr"],
                "time": trade["time"],
                "vwap_distance": abs(candle["close"] - candle["vwap"]),
                "candle_size": candle["high"] - candle["low"],
                "atr": candle["atr"],
                "result": result
            })

        trade_log_df = pd.DataFrame(trade_logs)
        trade_log_df.to_csv(output_path, index=False)

        print(f"Backtest completed. Trades saved to {output_path}")
        print(trade_log_df["result"].value_counts())
