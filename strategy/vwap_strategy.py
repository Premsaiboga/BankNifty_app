import pandas as pd


class VWAPPullbackStrategy:
    def __init__(self, max_trades_per_day=3, rr_list=[2, 3, 4]):
        self.max_trades_per_day = max_trades_per_day
        self.rr_list = rr_list

    # ===============================
    # DAILY VWAP (MANDATORY)
    # ===============================
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        VWAP PROXY for INDEX (no real volume available)
        Uses cumulative mean of Typical Price per session
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["datetime"]).dt.date

        vwap = pd.Series(index=df.index, dtype="float64")

        for date, idx in df.groupby("date").groups.items():
            day_df = df.loc[idx]

            typical_price = (
                day_df["high"] + day_df["low"] + day_df["close"]
            ) / 3

            session_vwap = typical_price.expanding().mean()
            vwap.loc[idx] = session_vwap

        return vwap

    @staticmethod
    def is_bullish(c):
        return c["close"] > c["open"]

    @staticmethod
    def is_bearish(c):
        return c["close"] < c["open"]

    # ===============================
    # CORE STRATEGY LOGIC (DEBUG MODE)
    # ===============================
    def generate_signals(self, df: pd.DataFrame):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["vwap"] = self.calculate_vwap(df)

        # Debug preview
        # print(df[["datetime", "close", "vwap"]].head(10))
        # print(df[["datetime", "close", "vwap"]].tail(10))

        signals = []
        trades_per_day = {}

        debug = {
            "total_candles": 0,
            "vwap_side_ok": 0,
            "vwap_proximity_ok": 0,
            "candle_ok": 0
        }

        for i in range(1, len(df)):
            curr = df.iloc[i]
            day = curr["date"]

            trades_per_day.setdefault(day, 0)

            if trades_per_day[day] >= self.max_trades_per_day:
                continue

            debug["total_candles"] += 1

            # VERY BASIC DEBUG CONDITION
            if curr["close"] > curr["vwap"]:
                debug["vwap_side_ok"] += 1

                entry = curr["close"]
                sl = curr["low"]

                if entry <= sl:
                    continue

                target = entry + (entry - sl) * 2

                signals.append({
                    "type": "BUY",
                    "entry": round(entry, 2),
                    "stoploss": round(sl, 2),
                    "target": round(target, 2),
                    "rr": 2,
                    "time": curr["datetime"]
                })

                trades_per_day[day] += 1

        return signals
