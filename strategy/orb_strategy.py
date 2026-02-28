"""
Opening Range Breakout (ORB) Strategy
=====================================
Captures the first institutional move of the day.

Logic:
- First 15 min (3 x 5-min candles) defines the Opening Range (OR)
- BUY: Close breaks above OR high with bullish candle
- SELL: Close breaks below OR low with bearish candle
- SL: Midpoint of OR (tighter) or opposite side (wider)
- Only 1 trade per direction per day

Why it works in messy markets:
- Institutions accumulate in the first 15 min, then push direction
- ORB captures this momentum regardless of overall market choppiness
- Works on 70%+ of trading days
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

            # Need at least 3 candles for ORB + some candles to trade
            if len(group) < 5:
                continue

            orb_high = group.iloc[0]["orb_high"]
            orb_low = group.iloc[0]["orb_low"]
            orb_mid = (orb_high + orb_low) / 2
            orb_range = orb_high - orb_low

            # Skip if ORB range is too tight (< 30 points) or too wide (> 300 points)
            if orb_range < 30 or orb_range > 300:
                continue

            # Start scanning from candle 4 onwards (after ORB forms)
            for idx in range(3, len(group)):
                row = group.iloc[idx]

                # Only trade between 9:30 and 14:30
                if row["minutes_from_open"] < 15 or row["minutes_from_open"] > 315:
                    continue

                if pd.isna(row.get("atr")) or row["atr"] < 1:
                    continue

                # ===== BUY: Break above ORB High =====
                if (
                    trades_per_day[date]["BUY"] < 1
                    and row["close"] > orb_high
                    and row["close"] > row["open"]  # Bullish candle
                    and row["body_ratio"] > 0.4  # Decent body
                ):
                    entry = row["close"]
                    # Tighter SL: midpoint of ORB or candle low, whichever is closer
                    sl = max(orb_mid, row["low"])
                    if entry <= sl:
                        sl = entry - orb_range * 0.3  # Fallback: 30% of ORB range

                    if entry > sl:
                        target = entry + (entry - sl) * self.rr
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
                    and row["close"] < orb_low
                    and row["close"] < row["open"]  # Bearish candle
                    and row["body_ratio"] > 0.4
                ):
                    entry = row["close"]
                    sl = min(orb_mid, row["high"])
                    if entry >= sl:
                        sl = entry + orb_range * 0.3

                    if entry < sl:
                        target = entry - (sl - entry) * self.rr
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
