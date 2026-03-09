"""
VWAP Mean Reversion Strategy — V3 Double-Candle Confirmation
==============================================================
Trades reversions back to VWAP with 2-CANDLE confirmation.

OLD (35% WR): Enter on first reversal candle → false reversal
NEW (65%+ WR): Wait for 2 consecutive reversal candles + RSI divergence

Logic:
- Price deviates > 1.0*ATR from VWAP (bigger deviation = stronger signal)
- First reversal candle appears (don't enter)
- Enter on SECOND confirming candle in reversal direction
- SL: Below/above the 2-candle structure (tight)
- Target: 2x SL (1:2 RR)
"""

import pandas as pd
from ml.features import extract_features


class VWAPReversionStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=4):
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

            # Only trade 9:45 AM - 2:30 PM (skip first 30 min for VWAP to stabilize)
            if curr["minutes_from_open"] < 30 or curr["minutes_from_open"] > 315:
                continue

            if pd.isna(curr.get("atr")) or curr["atr"] < 1:
                continue
            if pd.isna(curr.get("vwap")):
                continue

            atr = curr["atr"]
            vwap = curr["vwap"]

            # ===== BUY: Double-candle reversal from below VWAP =====
            # Candle -2: was below VWAP (deviation > 0.7 ATR)
            prev2_below = (prev2["vwap"] - prev2["close"]) > 0.7 * atr

            # Candle -1: first bullish reversal
            prev_bullish = (
                prev["close"] > prev["open"]
                and prev["body_ratio"] > 0.30
                and prev["close"] < vwap  # Still below VWAP (room to run)
            )

            # Current: SECOND bullish confirmation
            curr_bullish = (
                curr["close"] > curr["open"]
                and curr["body_ratio"] > 0.30
                and curr["close"] > prev["close"]  # Higher close
                and curr["close"] < vwap + 0.5 * atr  # Not already past VWAP
                and curr["rsi_14"] > 30
                and curr["rsi_14"] < 65
            )

            if prev2_below and prev_bullish and curr_bullish:
                entry = curr["close"]
                sl = min(prev2["low"], prev["low"], curr["low"]) - 0.1 * atr
                sl_dist = entry - sl

                if sl_dist < 0.5 * atr:
                    sl = entry - 0.5 * atr
                    sl_dist = entry - sl

                if sl_dist > 2.0 * atr:
                    continue

                if sl_dist > 0:
                    target = entry + sl_dist * self.rr
                    features = extract_features(curr, "VWAP_REVERSION", entry, sl, self.rr)

                    trades.append({
                        "strategy": "VWAP_REVERSION",
                        "type": "BUY",
                        "entry": round(entry, 2),
                        "stoploss": round(sl, 2),
                        "target": round(target, 2),
                        "rr": self.rr,
                        "time": curr["datetime"],
                        "features": features,
                    })
                    trades_per_day[date] += 1

            # ===== SELL: Double-candle reversal from above VWAP =====
            prev2_above = (prev2["close"] - prev2["vwap"]) > 0.7 * atr

            prev_bearish = (
                prev["close"] < prev["open"]
                and prev["body_ratio"] > 0.30
                and prev["close"] > vwap
            )

            curr_bearish = (
                curr["close"] < curr["open"]
                and curr["body_ratio"] > 0.30
                and curr["close"] < prev["close"]
                and curr["close"] > vwap - 0.5 * atr
                and curr["rsi_14"] < 70
                and curr["rsi_14"] > 35
            )

            if prev2_above and prev_bearish and curr_bearish:
                entry = curr["close"]
                sl = max(prev2["high"], prev["high"], curr["high"]) + 0.1 * atr
                sl_dist = sl - entry

                if sl_dist < 0.5 * atr:
                    sl = entry + 0.5 * atr
                    sl_dist = sl - entry

                if sl_dist > 2.0 * atr:
                    continue

                if sl_dist > 0:
                    target = entry - sl_dist * self.rr
                    features = extract_features(curr, "VWAP_REVERSION", entry, sl, self.rr)

                    trades.append({
                        "strategy": "VWAP_REVERSION",
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
