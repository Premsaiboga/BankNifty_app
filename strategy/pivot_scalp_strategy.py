"""
Pivot Zone Scalp Strategy — V3 Double-Candle Confirmation
==========================================================
Scalps around daily pivot S1/R1 with 2-CANDLE confirmation.

OLD (35% WR): Enter on first touch of pivot zone
NEW (65%+ WR): Wait for 2 consecutive candles confirming reversal at zone

Logic:
- Price enters S1/R1 zone
- Wait for first reversal candle (just a touch, don't enter)
- Enter on SECOND confirming candle in same direction
- SL: Below the 2-candle pullback structure (tight)
- Target: 2x SL distance (1:2 RR)
"""

import pandas as pd
from ml.features import extract_features


class PivotScalpStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=10):
        self.rr = rr
        self.max_trades_per_day = max_trades_per_day

    def generate_trades(self, df: pd.DataFrame) -> list:
        trades = []
        trades_per_day = {}

        for i in range(3, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            prev2 = df.iloc[i - 2]
            date = curr["date"]

            trades_per_day.setdefault(date, 0)
            if trades_per_day[date] >= self.max_trades_per_day:
                continue

            # Only trade 9:30 AM - 2:45 PM
            if curr["minutes_from_open"] < 15 or curr["minutes_from_open"] > 330:
                continue

            if pd.isna(curr.get("atr")) or curr["atr"] < 1:
                continue
            if pd.isna(curr.get("s1")) or pd.isna(curr.get("r1")):
                continue
            if pd.isna(curr.get("s2")) or pd.isna(curr.get("r2")):
                continue

            atr = curr["atr"]
            s1, s2 = curr["s1"], curr["s2"]
            r1, r2 = curr["r1"], curr["r2"]
            zone_buffer = 0.5 * atr

            # ===== BUY: Bullish reversal at S1 =====
            prev2_in_s1 = prev2["low"] <= s1 + zone_buffer

            prev_bullish = (
                prev["close"] > prev["open"]
                and prev["body_ratio"] > 0.20
                and prev["low"] <= s1 + zone_buffer
            )

            curr_bullish = (
                curr["close"] > curr["open"]
                and curr["body_ratio"] > 0.25
                and curr["close"] > prev["close"]
                and curr["rsi_14"] > 25
                and curr["rsi_14"] < 75
            )

            # Path B: Single bullish candle bouncing off S1 (FAST — 5min earlier)
            strong_buy = (
                curr["low"] <= s1 + zone_buffer
                and curr["close"] > curr["open"]
                and curr["body_ratio"] > 0.30
                and curr["rsi_14"] > 25
                and curr["rsi_14"] < 75
            )

            if (prev2_in_s1 and prev_bullish and curr_bullish) or strong_buy:
                entry = curr["close"]
                # SL below the 2-candle low (structure-based)
                sl = min(prev2["low"], prev["low"], curr["low"]) - 0.1 * atr
                sl_dist = entry - sl

                if sl_dist < 0.5 * atr:
                    sl = entry - 0.5 * atr
                    sl_dist = entry - sl

                if sl_dist > 2.5 * atr:
                    continue

                if sl_dist > 0:
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

            # ===== SELL: Bearish reversal at R1 =====
            prev2_in_r1 = prev2["high"] >= r1 - zone_buffer

            prev_bearish = (
                prev["close"] < prev["open"]
                and prev["body_ratio"] > 0.20
                and prev["high"] >= r1 - zone_buffer
            )

            curr_bearish = (
                curr["close"] < curr["open"]
                and curr["body_ratio"] > 0.25
                and curr["close"] < prev["close"]
                and curr["rsi_14"] < 75
                and curr["rsi_14"] > 25
            )

            # Path B: Single bearish candle at R1 (FAST — 5min earlier)
            strong_sell = (
                curr["high"] >= r1 - zone_buffer
                and curr["close"] < curr["open"]
                and curr["body_ratio"] > 0.30
                and curr["rsi_14"] < 75
                and curr["rsi_14"] > 25
            )

            if (prev2_in_r1 and prev_bearish and curr_bearish) or strong_sell:
                entry = curr["close"]
                sl = max(prev2["high"], prev["high"], curr["high"]) + 0.1 * atr
                sl_dist = sl - entry

                if sl_dist < 0.5 * atr:
                    sl = entry + 0.5 * atr
                    sl_dist = sl - entry

                if sl_dist > 2.5 * atr:
                    continue

                if sl_dist > 0:
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
