"""
EMA Crossover Scalp Strategy
=============================
Momentum trades on EMA crossovers with strong RSI confirmation.

Logic:
- BUY: 9 EMA crosses above 21 EMA + RSI > 55 + close > VWAP + strong candle
- SELL: 9 EMA crosses below 21 EMA + RSI < 45 + close < VWAP + strong candle
- SL: Minimum 1.0 ATR from entry
- RR 1.5
"""

import pandas as pd
from ml.features import extract_features


class EMAScalpStrategy:
    def __init__(self, rr=1.5, max_trades_per_day=2):
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

            if curr["minutes_from_open"] < 15 or curr["minutes_from_open"] > 330:
                continue

            if pd.isna(curr.get("atr")) or curr["atr"] < 1:
                continue
            if pd.isna(curr.get("rsi_14")):
                continue

            atr = curr["atr"]

            # FAKE BREAKOUT FILTERS
            consolidation = curr.get("consolidation_ratio", 2.0)
            ema_spread_abs = abs(curr.get("ema_spread", 0.0))
            range_vs_avg = curr.get("range_vs_avg", 1.0)

            # Skip if market is consolidating (EMAs will whipsaw)
            if consolidation < 1.2:
                continue

            # ===== EMA CROSS UP (BUY) =====
            ema_crossed_up = prev["ema_9"] <= prev["ema_21"] and curr["ema_9"] > curr["ema_21"]

            if (
                ema_crossed_up
                and curr["rsi_14"] > 55               # Strong momentum
                and curr["close"] > curr["vwap"]
                and curr["close"] > curr["open"]       # Bullish candle
                and curr["body_ratio"] > 0.45          # Decent body
                and ema_spread_abs > 0.15              # EMAs actually separating (not noise)
                and range_vs_avg > 1.0                 # Crossover candle is decent sized
            ):
                entry = curr["close"]
                # SL = LOWER of (EMA_21, candle_low) - buffer  [FIXED: was max()]
                sl = min(curr["ema_21"], curr["low"]) - 0.15 * atr
                sl_dist = entry - sl

                # Enforce minimum 1.0 ATR SL
                if sl_dist < 1.0 * atr:
                    sl = entry - 1.0 * atr
                    sl_dist = entry - sl

                # Skip if SL too wide (> 2.5 ATR)
                if sl_dist > 2.5 * atr:
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
                and curr["rsi_14"] < 45                # Strong weakness
                and curr["close"] < curr["vwap"]
                and curr["close"] < curr["open"]       # Bearish candle
                and curr["body_ratio"] > 0.45          # Decent body
                and ema_spread_abs > 0.15              # EMAs actually separating
                and range_vs_avg > 1.0                 # Crossover candle is decent sized
            ):
                entry = curr["close"]
                # SL = HIGHER of (EMA_21, candle_high) + buffer  [FIXED: was min()]
                sl = max(curr["ema_21"], curr["high"]) + 0.15 * atr
                sl_dist = sl - entry

                # Enforce minimum 1.0 ATR SL
                if sl_dist < 1.0 * atr:
                    sl = entry + 1.0 * atr
                    sl_dist = sl - entry

                if sl_dist > 2.5 * atr:
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
