"""
Momentum Surge Strategy
========================
Catches strong institutional candles and trades the continuation.

Logic:
- Detect a SURGE candle: body > 62% of range, range > 0.9*ATR
- BUY: Bullish surge + RSI > 55 + above VWAP
- SELL: Bearish surge + RSI < 45 + below VWAP
- SL: Beyond surge candle extreme (minimum 1.0 ATR)
- Target: Continuation at RR 2.0
"""

import pandas as pd
from ml.features import extract_features


class MomentumSurgeStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=2):
        self.rr = rr
        self.max_trades_per_day = max_trades_per_day

    def generate_trades(self, df: pd.DataFrame) -> list:
        trades = []
        trades_per_day = {}

        for i in range(1, len(df)):
            curr = df.iloc[i]
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
            candle_range = curr["candle_range"]

            # FAKE BREAKOUT FILTERS
            consolidation = curr.get("consolidation_ratio", 2.0)
            range_vs_avg = curr.get("range_vs_avg", 1.0)

            # Skip if market is consolidating (surges in chop are fake)
            if consolidation < 1.3:
                continue

            # ===== Surge Detection (TIGHTENED + range_vs_avg) =====
            is_surge = (
                curr["body_ratio"] > 0.62      # Strong institutional body
                and candle_range > 0.9 * atr    # Large candle
                and range_vs_avg > 1.5          # MUST be 1.5x bigger than recent candles
            )

            if not is_surge:
                continue

            # ===== BULLISH SURGE (BUY) =====
            if (
                curr["close"] > curr["open"]   # Bullish
                and curr["rsi_14"] > 55         # Strong momentum (was 50)
                and curr["close"] > curr.get("vwap", curr["close"])
            ):
                entry = curr["close"]
                sl = curr["low"] - 0.1 * atr   # ATR-scaled buffer (was -3)
                sl_dist = entry - sl

                # Enforce minimum 1.0 ATR SL
                if sl_dist < 1.0 * atr:
                    sl = entry - 1.0 * atr
                    sl_dist = entry - sl

                if sl_dist > 2.5 * atr:
                    continue

                target = entry + sl_dist * self.rr
                features = extract_features(curr, "MOMENTUM_SURGE", entry, sl, self.rr)

                trades.append({
                    "strategy": "MOMENTUM_SURGE",
                    "type": "BUY",
                    "entry": round(entry, 2),
                    "stoploss": round(sl, 2),
                    "target": round(target, 2),
                    "rr": self.rr,
                    "time": curr["datetime"],
                    "features": features,
                })
                trades_per_day[date] += 1

            # ===== BEARISH SURGE (SELL) =====
            elif (
                curr["close"] < curr["open"]   # Bearish
                and curr["rsi_14"] < 45         # Strong weakness (was 50)
                and curr["close"] < curr.get("vwap", curr["close"])
            ):
                entry = curr["close"]
                sl = curr["high"] + 0.1 * atr   # ATR-scaled buffer (was +3)
                sl_dist = sl - entry

                # Enforce minimum 1.0 ATR SL
                if sl_dist < 1.0 * atr:
                    sl = entry + 1.0 * atr
                    sl_dist = sl - entry

                if sl_dist > 2.5 * atr:
                    continue

                target = entry - sl_dist * self.rr
                features = extract_features(curr, "MOMENTUM_SURGE", entry, sl, self.rr)

                trades.append({
                    "strategy": "MOMENTUM_SURGE",
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
