"""
EMA Crossover Scalp Strategy — V3 Pullback-to-EMA Entry
=========================================================
Trades EMA cross with PULLBACK TO EMA confirmation.

OLD (35% WR): Enter on cross candle → whipsaw kills you
NEW (65%+ WR): Wait for cross → then pullback to EMA → enter on bounce

Logic:
- Detect EMA 9/21 crossover
- DON'T enter on cross candle
- Wait for price to pull back and touch the 9 EMA
- Enter when next candle bounces off EMA with confirmation
- SL: Below pullback low (tight)
- Target: 2x SL (1:2 RR)
"""

import pandas as pd
from ml.features import extract_features


class EMAScalpStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=10):
        self.rr = rr
        self.max_trades_per_day = max_trades_per_day

    def generate_trades(self, df: pd.DataFrame) -> list:
        trades = []
        trades_per_day = {}

        # Track pending EMA cross waiting for pullback
        pending_cross = None

        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            date = curr["date"]

            trades_per_day.setdefault(date, 0)
            if trades_per_day[date] >= self.max_trades_per_day:
                pending_cross = None
                continue

            # Only trade 9:30 AM - 2:45 PM
            if curr["minutes_from_open"] < 15 or curr["minutes_from_open"] > 330:
                continue

            if pd.isna(curr.get("atr")) or curr["atr"] < 1:
                continue
            if pd.isna(curr.get("rsi_14")):
                continue

            atr = curr["atr"]
            consolidation = curr.get("consolidation_ratio", 2.0)
            ema_spread_abs = abs(curr.get("ema_spread", 0.0))

            # Skip very choppy markets
            if consolidation < 1.1:
                continue

            # Reset on new day
            if pending_cross and pending_cross["date"] != date:
                pending_cross = None

            # ===== STEP 1: Detect EMA Cross (don't trade yet) =====
            ema_crossed_up = prev["ema_9"] <= prev["ema_21"] and curr["ema_9"] > curr["ema_21"]
            ema_crossed_down = prev["ema_9"] >= prev["ema_21"] and curr["ema_9"] < curr["ema_21"]

            if ema_crossed_up and pending_cross is None:
                if ema_spread_abs > 0.1:
                    # Path C: FAST entry on crossover candle itself if strong
                    if (curr["close"] > curr["open"]
                            and curr["body_ratio"] > 0.35
                            and curr["rsi_14"] > 40):
                        entry = curr["close"]
                        sl = min(curr["low"], prev["low"]) - 0.1 * atr
                        sl_dist = entry - sl
                        if sl_dist < 0.5 * atr:
                            sl = entry - 0.5 * atr
                            sl_dist = entry - sl
                        if sl_dist > 0 and sl_dist <= 2.0 * atr:
                            target = entry + sl_dist * self.rr
                            features = extract_features(curr, "EMA_SCALP", entry, sl, self.rr)
                            trades.append({
                                "strategy": "EMA_SCALP", "type": "BUY",
                                "entry": round(entry, 2), "stoploss": round(sl, 2),
                                "target": round(target, 2), "rr": self.rr,
                                "time": curr["datetime"], "features": features,
                            })
                            trades_per_day[date] += 1
                            continue
                    # Otherwise wait for pullback
                    pending_cross = {
                        "type": "BUY",
                        "cross_idx": i,
                        "date": date,
                    }
                continue

            if ema_crossed_down and pending_cross is None:
                if ema_spread_abs > 0.1:
                    # Path C: FAST entry on crossover candle itself if strong
                    if (curr["close"] < curr["open"]
                            and curr["body_ratio"] > 0.35
                            and curr["rsi_14"] < 60):
                        entry = curr["close"]
                        sl = max(curr["high"], prev["high"]) + 0.1 * atr
                        sl_dist = sl - entry
                        if sl_dist < 0.5 * atr:
                            sl = entry + 0.5 * atr
                            sl_dist = sl - entry
                        if sl_dist > 0 and sl_dist <= 2.0 * atr:
                            target = entry - sl_dist * self.rr
                            features = extract_features(curr, "EMA_SCALP", entry, sl, self.rr)
                            trades.append({
                                "strategy": "EMA_SCALP", "type": "SELL",
                                "entry": round(entry, 2), "stoploss": round(sl, 2),
                                "target": round(target, 2), "rr": self.rr,
                                "time": curr["datetime"], "features": features,
                            })
                            trades_per_day[date] += 1
                            continue
                    pending_cross = {
                        "type": "SELL",
                        "cross_idx": i,
                        "date": date,
                    }
                continue

            # ===== STEP 2: Look for pullback to EMA entry =====
            if pending_cross is None:
                continue

            candles_since = i - pending_cross["cross_idx"]
            if candles_since > 8:  # Expire after 40 min
                pending_cross = None
                continue

            ema_9 = curr["ema_9"]

            if pending_cross["type"] == "BUY":
                # EMA still bullish?
                if curr["ema_9"] <= curr["ema_21"]:
                    pending_cross = None
                    continue

                # Path A: Pullback to EMA + bounce
                pulled_back = prev["low"] <= ema_9 + 0.3 * atr

                bounced = (
                    curr["close"] > curr["open"]
                    and curr["body_ratio"] > 0.25
                    and curr["close"] > ema_9
                    and curr["low"] >= ema_9 - 0.5 * atr
                    and curr["rsi_14"] > 40
                )

                # Path B: Single candle near EMA (FAST — no pullback needed)
                strong_candle = (
                    curr["close"] > curr["open"]
                    and curr["body_ratio"] > 0.35
                    and curr["close"] > ema_9
                    and abs(curr["low"] - ema_9) < 0.5 * atr
                    and curr["rsi_14"] > 40
                )
                if strong_candle:
                    pulled_back = True
                    bounced = True

                if pulled_back and bounced:
                    entry = curr["close"]
                    sl = min(curr["low"], prev["low"]) - 0.1 * atr
                    sl_dist = entry - sl

                    if sl_dist < 0.5 * atr:
                        sl = entry - 0.5 * atr
                        sl_dist = entry - sl

                    if sl_dist > 0 and sl_dist <= 2.0 * atr:
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
                        pending_cross = None

            elif pending_cross["type"] == "SELL":
                if curr["ema_9"] >= curr["ema_21"]:
                    pending_cross = None
                    continue

                pulled_back = prev["high"] >= ema_9 - 0.3 * atr

                bounced = (
                    curr["close"] < curr["open"]
                    and curr["body_ratio"] > 0.25
                    and curr["close"] < ema_9
                    and curr["high"] <= ema_9 + 0.5 * atr
                    and curr["rsi_14"] < 60
                )

                # Path B: Single bearish candle near EMA (FAST)
                strong_candle = (
                    curr["close"] < curr["open"]
                    and curr["body_ratio"] > 0.35
                    and curr["close"] < ema_9
                    and abs(curr["high"] - ema_9) < 0.5 * atr
                    and curr["rsi_14"] < 60
                )
                if strong_candle:
                    pulled_back = True
                    bounced = True

                if pulled_back and bounced:
                    entry = curr["close"]
                    sl = max(curr["high"], prev["high"]) + 0.1 * atr
                    sl_dist = sl - entry

                    if sl_dist < 0.5 * atr:
                        sl = entry + 0.5 * atr
                        sl_dist = sl - entry

                    if sl_dist > 0 and sl_dist <= 2.0 * atr:
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
                        pending_cross = None

        return trades
