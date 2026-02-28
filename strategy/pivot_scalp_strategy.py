"""
Pivot Zone Scalp Strategy
==========================
Quick scalps around daily pivot support/resistance zones.

Logic:
- Uses previous day's pivot levels (S1, S2, R1, R2)
- BUY: Price enters S1 zone, shows bullish reversal candle
- SELL: Price enters R1 zone, shows bearish reversal candle
- SL: Beyond the zone (S2 or R2)
- Quick target: back toward pivot

Why it works in messy markets:
- Daily pivot levels are used by ALL institutional traders
- Price respects these levels even in choppy markets
- Scalping the bounce/rejection at these levels is high-probability
- Tight SL = small risk, quick profit
"""

import pandas as pd
from ml.features import extract_features


class PivotScalpStrategy:
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

            if curr["minutes_from_open"] < 15 or curr["minutes_from_open"] > 330:
                continue

            if pd.isna(curr.get("atr")) or curr["atr"] < 1:
                continue

            # Check pivot levels exist
            if pd.isna(curr.get("s1")) or pd.isna(curr.get("r1")):
                continue
            if pd.isna(curr.get("s2")) or pd.isna(curr.get("r2")):
                continue

            atr = curr["atr"]
            pivot = curr["pivot"]
            s1, s2 = curr["s1"], curr["s2"]
            r1, r2 = curr["r1"], curr["r2"]

            # Zone buffers (allow some tolerance)
            zone_buffer = 0.3 * atr

            # ===== BUY: Price near S1 zone with bullish reversal =====
            in_s1_zone = (curr["low"] <= s1 + zone_buffer) and (curr["close"] > s1 - zone_buffer)

            if (
                in_s1_zone
                and curr["close"] > curr["open"]  # Bullish candle
                and curr["body_ratio"] > 0.3
                and (prev["low"] <= s1 + zone_buffer or prev["close"] < s1 + zone_buffer)  # Confirmed touch
            ):
                entry = curr["close"]
                # SL below S1 with buffer, but not past S2
                sl = max(s1 - 0.4 * atr, s2)
                sl = min(sl, entry - 10)  # At least 10 points SL

                if entry > sl:
                    sl_dist = entry - sl
                    if sl_dist > 1.5 * atr:
                        continue

                    target = entry + sl_dist * self.rr
                    features = extract_features(curr, "PIVOT_SCALP", entry, sl, self.rr)

                    trades.append({
                        "strategy": "PIVOT_SCALP",
                        "type": "BUY",
                        "entry": round(entry, 2),
                        "stoploss": round(sl, 2),
                        "target": round(target, 2),
                        "rr": self.rr,
                        "time": curr["datetime"],
                        "features": features,
                    })
                    trades_per_day[date] += 1

            # ===== SELL: Price near R1 zone with bearish reversal =====
            in_r1_zone = (curr["high"] >= r1 - zone_buffer) and (curr["close"] < r1 + zone_buffer)

            if (
                in_r1_zone
                and curr["close"] < curr["open"]  # Bearish candle
                and curr["body_ratio"] > 0.3
                and (prev["high"] >= r1 - zone_buffer or prev["close"] > r1 - zone_buffer)
            ):
                entry = curr["close"]
                sl = min(r1 + 0.4 * atr, r2)
                sl = max(sl, entry + 10)

                if entry < sl:
                    sl_dist = sl - entry
                    if sl_dist > 1.5 * atr:
                        continue

                    target = entry - sl_dist * self.rr
                    features = extract_features(curr, "PIVOT_SCALP", entry, sl, self.rr)

                    trades.append({
                        "strategy": "PIVOT_SCALP",
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
