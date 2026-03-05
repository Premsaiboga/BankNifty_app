"""
Momentum Surge Strategy — V3 Pullback Entry
=============================================
Catches institutional surges and enters on PULLBACK, not on the surge itself.

OLD (35% WR): Enter on surge candle → already extended, gets reversed
NEW (65%+ WR): Detect surge → wait for 38-50% pullback → enter on bounce

Logic:
- Detect SURGE candle (body>62%, range>1.2*ATR, range_vs_avg>1.5)
- DON'T enter — wait for next candle(s) to pull back
- Enter when pullback holds at 38-50% of surge and resumes
- SL: Below pullback low (tight)
- Target: 2x SL distance (1:2 RR)
"""

import pandas as pd
import numpy as np
from ml.features import extract_features


class MomentumSurgeStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=2):
        self.rr = rr
        self.max_trades_per_day = max_trades_per_day

    def generate_trades(self, df: pd.DataFrame) -> list:
        trades = []
        trades_per_day = {}

        # Track pending surges waiting for pullback
        pending_surge = None

        for i in range(1, len(df)):
            curr = df.iloc[i]
            date = curr["date"]

            trades_per_day.setdefault(date, 0)
            if trades_per_day[date] >= self.max_trades_per_day:
                pending_surge = None
                continue

            # Only trade 9:30 AM - 2:30 PM
            if curr["minutes_from_open"] < 15 or curr["minutes_from_open"] > 315:
                continue

            if pd.isna(curr.get("atr")) or curr["atr"] < 1:
                continue
            if pd.isna(curr.get("rsi_14")):
                continue

            atr = curr["atr"]
            candle_range = curr["candle_range"]
            consolidation = curr.get("consolidation_ratio", 2.0)
            range_vs_avg = curr.get("range_vs_avg", 1.0)

            # Reset pending surge on new day
            if pending_surge and pending_surge["date"] != date:
                pending_surge = None

            # ===== STEP 1: Detect Surge (don't trade yet) =====
            is_surge = (
                curr["body_ratio"] > 0.62
                and candle_range > 1.2 * atr  # STRONGER surge required (was 0.9)
                and range_vs_avg > 1.5
                and consolidation > 1.3
            )

            if is_surge and pending_surge is None:
                if curr["close"] > curr["open"]:
                    pending_surge = {
                        "type": "BUY",
                        "surge_high": curr["high"],
                        "surge_low": curr["low"],
                        "surge_close": curr["close"],
                        "surge_open": curr["open"],
                        "surge_idx": i,
                        "date": date,
                    }
                elif curr["close"] < curr["open"]:
                    pending_surge = {
                        "type": "SELL",
                        "surge_high": curr["high"],
                        "surge_low": curr["low"],
                        "surge_close": curr["close"],
                        "surge_open": curr["open"],
                        "surge_idx": i,
                        "date": date,
                    }
                continue

            # ===== STEP 2: Look for pullback entry =====
            if pending_surge is None:
                continue

            candles_since = i - pending_surge["surge_idx"]
            # Expire after 4 candles (20 min) — pullback should be quick
            if candles_since > 4:
                pending_surge = None
                continue

            surge_range = pending_surge["surge_high"] - pending_surge["surge_low"]
            prev = df.iloc[i - 1]

            if pending_surge["type"] == "BUY":
                # Pullback: price dipped into 38-62% of surge range
                pullback_level = pending_surge["surge_high"] - 0.38 * surge_range
                deep_level = pending_surge["surge_high"] - 0.62 * surge_range

                pulled_back = prev["low"] <= pullback_level
                held_support = prev["low"] >= deep_level  # Didn't break too deep

                # Bounce: current candle is bullish and closes above pullback
                bounced = (
                    curr["close"] > curr["open"]
                    and curr["body_ratio"] > 0.40
                    and curr["close"] > pullback_level
                    and curr["rsi_14"] > 48
                    and curr["close"] > curr.get("vwap", curr["close"])
                )

                if pulled_back and held_support and bounced:
                    entry = curr["close"]
                    sl = min(curr["low"], prev["low"]) - 0.1 * atr
                    sl_dist = entry - sl

                    if sl_dist < 0.5 * atr:
                        sl = entry - 0.5 * atr
                        sl_dist = entry - sl

                    if sl_dist > 0 and sl_dist <= 2.0 * atr:
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
                        pending_surge = None

            elif pending_surge["type"] == "SELL":
                pullback_level = pending_surge["surge_low"] + 0.38 * surge_range
                deep_level = pending_surge["surge_low"] + 0.62 * surge_range

                pulled_back = prev["high"] >= pullback_level
                held_resistance = prev["high"] <= deep_level

                bounced = (
                    curr["close"] < curr["open"]
                    and curr["body_ratio"] > 0.40
                    and curr["close"] < pullback_level
                    and curr["rsi_14"] < 52
                    and curr["close"] < curr.get("vwap", curr["close"])
                )

                if pulled_back and held_resistance and bounced:
                    entry = curr["close"]
                    sl = max(curr["high"], prev["high"]) + 0.1 * atr
                    sl_dist = sl - entry

                    if sl_dist < 0.5 * atr:
                        sl = entry + 0.5 * atr
                        sl_dist = sl - entry

                    if sl_dist > 0 and sl_dist <= 2.0 * atr:
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
                        pending_surge = None

        return trades
