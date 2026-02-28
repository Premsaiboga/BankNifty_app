"""
EMA Crossover Scalp Strategy
=============================
Quick momentum trades on EMA crossovers with RSI confirmation.

Logic:
- BUY: 9 EMA crosses above 21 EMA + RSI > 50 + close > VWAP
- SELL: 9 EMA crosses below 21 EMA + RSI < 50 + close < VWAP
- SL: 21 EMA or recent candle low/high
- Quick scalp with RR 1.5

Why it works in messy markets:
- EMA crossovers catch momentum shifts even in chop
- RSI filter prevents trading in dead zones
- VWAP adds institutional flow direction
- Low RR means quick target hits
"""

import pandas as pd
from ml.features import extract_features


class EMAScalpStrategy:
    def __init__(self, rr=1.5, max_trades_per_day=3):
        self.rr = rr
        self.max_trades_per_day = max_trades_per_day

    def generate_trades(self, df: pd.DataFrame) -> list:
        trades = []
        trades_per_day = {}

        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            date = curr["date"]

            trades_per_day.setdefault(date, 0)
            if trades_per_day[date] >= self.max_trades_per_day:
                continue

            # Only trade between 9:30 and 14:45
            if curr["minutes_from_open"] < 15 or curr["minutes_from_open"] > 330:
                continue

            if pd.isna(curr.get("atr")) or curr["atr"] < 1:
                continue
            if pd.isna(curr.get("rsi_14")):
                continue

            # ===== EMA CROSS UP (BUY) =====
            ema_crossed_up = prev["ema_9"] <= prev["ema_21"] and curr["ema_9"] > curr["ema_21"]

            if (
                ema_crossed_up
                and curr["rsi_14"] > 48  # Slightly relaxed from 50
                and curr["close"] > curr["vwap"]
                and curr["close"] > curr["open"]  # Bullish candle
            ):
                entry = curr["close"]
                # SL = 21 EMA or candle low, whichever gives tighter SL
                sl = max(curr["ema_21"], curr["low"]) - 5  # 5-point buffer
                if entry <= sl:
                    continue

                sl_dist = entry - sl
                # Skip if SL too wide (> 1.5 ATR) or too tight (< 0.3 ATR)
                if sl_dist > 1.5 * curr["atr"] or sl_dist < 0.3 * curr["atr"]:
                    continue

                target = entry + sl_dist * self.rr
                features = extract_features(curr, "EMA_SCALP", entry, sl, self.rr)

                trades.append({
                    "strategy": "EMA_SCALP",
                    "type": "BUY",
                    "entry": round(entry, 2),
                    "stoploss": round(sl, 2),
                    "target": round(target, 2),
                    "rr": self.rr,
                    "time": curr["datetime"],
                    "features": features,
                })
                trades_per_day[date] += 1

            # ===== EMA CROSS DOWN (SELL) =====
            ema_crossed_down = prev["ema_9"] >= prev["ema_21"] and curr["ema_9"] < curr["ema_21"]

            if (
                ema_crossed_down
                and curr["rsi_14"] < 52  # Slightly relaxed
                and curr["close"] < curr["vwap"]
                and curr["close"] < curr["open"]  # Bearish candle
            ):
                entry = curr["close"]
                sl = min(curr["ema_21"], curr["high"]) + 5
                if entry >= sl:
                    continue

                sl_dist = sl - entry
                if sl_dist > 1.5 * curr["atr"] or sl_dist < 0.3 * curr["atr"]:
                    continue

                target = entry - sl_dist * self.rr
                features = extract_features(curr, "EMA_SCALP", entry, sl, self.rr)

                trades.append({
                    "strategy": "EMA_SCALP",
                    "type": "SELL",
                    "entry": round(entry, 2),
                    "stoploss": round(sl, 2),
                    "target": round(target, 2),
                    "rr": self.rr,
                    "time": curr["datetime"],
                    "features": features,
                })
                trades_per_day[date] += 1

        return trades
