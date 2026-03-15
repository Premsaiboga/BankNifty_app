"""
Opening Range Breakout (ORB) Strategy — V3 Pullback Entry
==========================================================
Captures institutional ORB moves with PULLBACK confirmation.

OLD (35% WR): Enter on breakout candle → gets faked out
NEW (65%+ WR): Wait for breakout → then retest → enter on bounce

Logic:
- First 15 min defines Opening Range (OR)
- Wait for DECISIVE close above/below OR
- Then wait for price to PULL BACK toward OR level
- Enter when pullback holds and price resumes direction
- SL: Below pullback low (tight, structure-based)
- Target: 2x SL distance (1:2 RR)
"""

import pandas as pd
import numpy as np
from ml.features import extract_features


class ORBStrategy:
    def __init__(self, rr=2.0, max_trades_per_day=6):
        self.rr = rr
        self.max_trades_per_day = max_trades_per_day

    def generate_trades(self, df: pd.DataFrame) -> list:
        trades = []
        trades_per_day = {}

        for date, group in df.groupby("date"):
            trades_per_day.setdefault(date, {"BUY": 0, "SELL": 0})

            if len(group) < 6:
                continue

            orb_high = group.iloc[0]["orb_high"]
            orb_low = group.iloc[0]["orb_low"]
            orb_range = orb_high - orb_low

            # Skip if ORB range is too tight (< 40) or too wide (> 250)
            if orb_range < 40 or orb_range > 250:
                continue

            # Track breakout state for this day
            broke_above = False
            broke_below = False
            breakout_candle_idx = None

            for idx in range(3, len(group)):
                row = group.iloc[idx]

                # Only trade between 9:30 and 15:00
                if row["minutes_from_open"] < 15 or row["minutes_from_open"] > 345:
                    continue

                if pd.isna(row.get("atr")) or row["atr"] < 1:
                    continue
                if pd.isna(row.get("rsi_14")):
                    continue

                atr = row["atr"]
                consolidation = row.get("consolidation_ratio", 2.0)

                # Skip very choppy markets
                if consolidation < 1.0:
                    continue

                # ===== DETECT BREAKOUT =====
                if not broke_above and row["close"] > orb_high + 0.2 * atr:
                    if row["close"] > row["open"] and row["body_ratio"] > 0.35:
                        broke_above = True
                        breakout_candle_idx = idx
                        # FAST PATH: Enter on breakout candle if strong enough
                        if row["body_ratio"] > 0.40 and row["rsi_14"] > 40 and trades_per_day[date]["BUY"] < 3:
                            prev_row = group.iloc[idx - 1] if idx > 0 else row
                            entry = row["close"]
                            sl = min(row["low"], prev_row["low"]) - 0.1 * atr
                            sl_dist = entry - sl
                            if sl_dist < 0.5 * atr:
                                sl = entry - 0.5 * atr
                                sl_dist = entry - sl
                            if sl_dist > 0 and sl_dist <= 2.5 * atr:
                                target = entry + sl_dist * self.rr
                                features = extract_features(row, "ORB", entry, sl, self.rr)
                                trades.append({
                                    "strategy": "ORB", "type": "BUY",
                                    "entry": round(entry, 2), "stoploss": round(sl, 2),
                                    "target": round(target, 2), "rr": self.rr,
                                    "time": row["datetime"], "features": features,
                                })
                                trades_per_day[date]["BUY"] += 1
                        continue

                if not broke_below and row["close"] < orb_low - 0.2 * atr:
                    if row["close"] < row["open"] and row["body_ratio"] > 0.35:
                        broke_below = True
                        breakout_candle_idx = idx
                        # FAST PATH: Enter on breakout candle if strong enough
                        if row["body_ratio"] > 0.40 and row["rsi_14"] < 60 and trades_per_day[date]["SELL"] < 3:
                            prev_row = group.iloc[idx - 1] if idx > 0 else row
                            entry = row["close"]
                            sl = max(row["high"], prev_row["high"]) + 0.1 * atr
                            sl_dist = sl - entry
                            if sl_dist < 0.5 * atr:
                                sl = entry + 0.5 * atr
                                sl_dist = sl - entry
                            if sl_dist > 0 and sl_dist <= 2.5 * atr:
                                target = entry - sl_dist * self.rr
                                features = extract_features(row, "ORB", entry, sl, self.rr)
                                trades.append({
                                    "strategy": "ORB", "type": "SELL",
                                    "entry": round(entry, 2), "stoploss": round(sl, 2),
                                    "target": round(target, 2), "rr": self.rr,
                                    "time": row["datetime"], "features": features,
                                })
                                trades_per_day[date]["SELL"] += 1
                        continue

                # ===== BUY: Pullback after upward breakout =====
                if (
                    broke_above
                    and trades_per_day[date]["BUY"] < 3
                    and breakout_candle_idx is not None
                    and idx > breakout_candle_idx
                    and idx <= breakout_candle_idx + 8  # Within 40 min of breakout
                ):
                    prev = group.iloc[idx - 1]

                    # Path A: Pullback then bounce
                    pulled_back = prev["low"] <= orb_high + 0.5 * atr
                    bounced = (
                        row["close"] > row["open"]
                        and row["close"] > orb_high
                        and row["body_ratio"] > 0.25
                        and row["rsi_14"] > 40
                    )

                    # Path B: Continuation candle above ORB high (FAST)
                    strong_cont = (
                        row["close"] > row["open"]
                        and row["close"] > orb_high
                        and row["body_ratio"] > 0.30
                        and row["rsi_14"] > 40
                    )
                    if strong_cont:
                        pulled_back = True
                        bounced = True

                    if pulled_back and bounced:
                        entry = row["close"]
                        # Tight SL: below pullback low (structure-based)
                        sl = min(row["low"], prev["low"]) - 0.1 * atr

                        sl_dist = entry - sl
                        if sl_dist < 0.5 * atr:
                            sl = entry - 0.5 * atr
                            sl_dist = entry - sl

                        if sl_dist > 0 and sl_dist <= 2.5 * atr:
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

                # ===== SELL: Pullback after downward breakout =====
                if (
                    broke_below
                    and trades_per_day[date]["SELL"] < 3
                    and breakout_candle_idx is not None
                    and idx > breakout_candle_idx
                    and idx <= breakout_candle_idx + 8
                ):
                    prev = group.iloc[idx - 1]

                    # Path A: Pullback then drop
                    pulled_back = prev["high"] >= orb_low - 0.5 * atr
                    bounced = (
                        row["close"] < row["open"]
                        and row["close"] < orb_low
                        and row["body_ratio"] > 0.25
                        and row["rsi_14"] < 60
                    )

                    # Path B: Continuation candle below ORB low (FAST)
                    strong_cont = (
                        row["close"] < row["open"]
                        and row["close"] < orb_low
                        and row["body_ratio"] > 0.30
                        and row["rsi_14"] < 60
                    )
                    if strong_cont:
                        pulled_back = True
                        bounced = True

                    if pulled_back and bounced:
                        entry = row["close"]
                        sl = max(row["high"], prev["high"]) + 0.1 * atr

                        sl_dist = sl - entry
                        if sl_dist < 0.5 * atr:
                            sl = entry + 0.5 * atr
                            sl_dist = sl - entry

                        if sl_dist > 0 and sl_dist <= 2.5 * atr:
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
