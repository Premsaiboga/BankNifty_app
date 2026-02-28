"""
Momentum Surge Strategy
========================
Catches strong institutional candles and trades the continuation.

Logic:
- Detect a SURGE candle: body > 60% of range, range > 0.8*ATR
- BUY: Bullish surge + RSI > 52 + candle close above VWAP
- SELL: Bearish surge + RSI < 48 + candle close below VWAP
- SL: Opposite end of the surge candle
- Target: Continuation move

Why it works in messy markets:
- Even in chop, institutions create sudden surges
- These surges have follow-through in the next 2-3 candles
- Large candles = institutional order flow, not retail noise
- Riding the surge gets quick profits before the next chop starts
"""

import pandas as pd
from ml.features import extract_features


class MomentumSurgeStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=3):
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

            # ===== Surge Detection =====
            is_surge = (
                curr["body_ratio"] > 0.55  # Strong body (relaxed from 0.6)
                and candle_range > 0.7 * atr  # Large candle (relaxed from 0.8)
            )

            if not is_surge:
                continue

            # ===== BULLISH SURGE (BUY) =====
            if (
                curr["close"] > curr["open"]  # Bullish
                and curr["rsi_14"] > 50
                and curr["close"] > curr.get("vwap", curr["close"])  # Above VWAP
            ):
                entry = curr["close"]
                sl = curr["low"] - 3  # SL at candle low with buffer

                sl_dist = entry - sl
                if sl_dist <= 0 or sl_dist > 2.0 * atr:
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
                curr["close"] < curr["open"]  # Bearish
                and curr["rsi_14"] < 50
                and curr["close"] < curr.get("vwap", curr["close"])  # Below VWAP
            ):
                entry = curr["close"]
                sl = curr["high"] + 3

                sl_dist = sl - entry
                if sl_dist <= 0 or sl_dist > 2.0 * atr:
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
