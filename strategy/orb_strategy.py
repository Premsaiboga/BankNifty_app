"""
Opening Range Breakout (ORB) Strategy
=====================================
Captures the first institutional move of the day.

Logic:
- First 15 min (3 x 5-min candles) defines the Opening Range (OR)
- BUY: Close breaks above OR high with bullish candle + RSI confirmation
- SELL: Close breaks below OR low with bearish candle + RSI confirmation
- SL: Midpoint of OR (minimum 1.0 ATR)
- Only 1 trade per direction per day
"""

import pandas as pd
from ml.features import extract_features


class ORBStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=2):
        self.rr = rr
        self.max_trades_per_day = max_trades_per_day

    def generate_trades(self, df: pd.DataFrame) -> list:
        trades = []
        trades_per_day = {}

        for date, group in df.groupby("date"):
            trades_per_day.setdefault(date, {"BUY": 0, "SELL": 0})

            if len(group) < 5:
                continue

            orb_high = group.iloc[0]["orb_high"]
            orb_low = group.iloc[0]["orb_low"]
            orb_mid = (orb_high + orb_low) / 2
            orb_range = orb_high - orb_low

            # Skip if ORB range is too tight (< 40) or too wide (> 250)
            if orb_range < 40 or orb_range > 250:
                continue

            for idx in range(3, len(group)):
                row = group.iloc[idx]

                # Only trade between 9:30 and 14:30
                if row["minutes_from_open"] < 15 or row["minutes_from_open"] > 315:
                    continue

                if pd.isna(row.get("atr")) or row["atr"] < 1:
                    continue
                if pd.isna(row.get("rsi_14")):
                    continue

                atr = row["atr"]

                # FAKE BREAKOUT FILTERS
                consolidation = row.get("consolidation_ratio", 2.0)
                range_vs_avg = row.get("range_vs_avg", 1.0)

                # Skip if market is consolidating (tight range = fake breakouts)
                if consolidation < 1.2:
                    continue

                # ===== BUY: Break above ORB High =====
                if (
                    trades_per_day[date]["BUY"] < 1
                    and row["close"] > orb_high + 0.3 * atr  # DECISIVE break (not barely above)
                    and row["close"] > row["open"]      # Bullish candle
                    and row["body_ratio"] > 0.50         # Strong body
                    and row["rsi_14"] > 55               # RSI confirms momentum
                    and row["close"] > row.get("vwap", row["close"])  # Above VWAP
                    and range_vs_avg > 1.2               # Breakout candle bigger than recent avg
                ):
                    entry = row["close"]
                    sl = max(orb_mid, row["low"])
                    if entry <= sl:
                        sl = entry - orb_range * 0.3

                    # Enforce minimum SL of 1.0 ATR
                    sl_dist = entry - sl
                    if sl_dist < 1.0 * atr:
                        sl = entry - 1.0 * atr
                    sl_dist = entry - sl

                    if sl_dist > 0 and sl_dist <= 3.0 * atr:
                        target = entry + sl_dist * self.rr
                        features = extract_features(row, "ORB", entry, sl, self.rr)

                        trades.append({
                            "strategy": "ORB",
                            "type": "BUY",
                            "entry": round(entry, 2),
                            "stoploss": round(sl, 2),
                            "target": round(target, 2),
                            "rr": self.rr,
                            "time": row["datetime"],
                            "features": features,
                        })
                        trades_per_day[date]["BUY"] += 1

                # ===== SELL: Break below ORB Low =====
                if (
                    trades_per_day[date]["SELL"] < 1
                    and row["close"] < orb_low - 0.3 * atr  # DECISIVE break (not barely below)
                    and row["close"] < row["open"]       # Bearish candle
                    and row["body_ratio"] > 0.50          # Strong body
                    and row["rsi_14"] < 45                # RSI confirms weakness
                    and row["close"] < row.get("vwap", row["close"])  # Below VWAP
                    and range_vs_avg > 1.2               # Breakout candle bigger than recent avg
                ):
                    entry = row["close"]
                    sl = min(orb_mid, row["high"])
                    if entry >= sl:
                        sl = entry + orb_range * 0.3

                    # Enforce minimum SL of 1.0 ATR
                    sl_dist = sl - entry
                    if sl_dist < 1.0 * atr:
                        sl = entry + 1.0 * atr
                    sl_dist = sl - entry

                    if sl_dist > 0 and sl_dist <= 3.0 * atr:
                        target = entry - sl_dist * self.rr
                        features = extract_features(row, "ORB", entry, sl, self.rr)

                        trades.append({
                            "strategy": "ORB",
                            "type": "SELL",
                            "entry": round(entry, 2),
                            "stoploss": round(sl, 2),
                            "target": round(target, 2),
                            "rr": self.rr,
                            "time": row["datetime"],
                            "features": features,
                        })
                        trades_per_day[date]["SELL"] += 1

        return trades
