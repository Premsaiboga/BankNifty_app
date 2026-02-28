"""
VWAP Mean Reversion Strategy
==============================
Trades reversions back to VWAP when price deviates too far.

Logic:
- BUY: Price drops below VWAP - 0.5*ATR, then shows bullish reversal candle
- SELL: Price rises above VWAP + 0.5*ATR, then shows bearish reversal candle
- SL: Beyond the deviation extreme (VWAP ± 1.5*ATR)
- Target: Back toward VWAP

Why it works in messy markets:
- Messy markets = mean-reverting price action around VWAP
- Institutions use VWAP for execution, price always reverts
- This is THE strategy for choppy institutional flow
- Low RR = quick hits as price snaps back to VWAP
"""

import pandas as pd
from ml.features import extract_features


class VWAPReversionStrategy:
    def __init__(self, rr=1.5, max_trades_per_day=4):
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
            if pd.isna(curr.get("vwap")):
                continue

            atr = curr["atr"]
            vwap = curr["vwap"]
            close = curr["close"]

            # ===== BUY: Price dropped below VWAP, reverting up =====
            deviation_down = vwap - close
            if (
                deviation_down > 0.4 * atr  # Price is below VWAP by at least 0.4 ATR
                and close > curr["open"]  # Current candle is bullish (reversal)
                and curr["body_ratio"] > 0.35  # Has decent body
                and prev["close"] < vwap  # Prev also below VWAP (confirmed deviation)
            ):
                entry = close
                # SL below the swing low, but capped at 1.2 ATR
                sl = min(curr["low"], prev["low"]) - 3
                sl_dist = entry - sl

                if sl_dist <= 0 or sl_dist > 1.2 * atr:
                    sl = entry - 0.8 * atr  # Cap SL
                    sl_dist = entry - sl

                if sl_dist > 0.2 * atr:  # Minimum SL distance
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

            # ===== SELL: Price rose above VWAP, reverting down =====
            deviation_up = close - vwap
            if (
                deviation_up > 0.4 * atr
                and close < curr["open"]  # Bearish candle (reversal)
                and curr["body_ratio"] > 0.35
                and prev["close"] > vwap
            ):
                entry = close
                sl = max(curr["high"], prev["high"]) + 3
                sl_dist = sl - entry

                if sl_dist <= 0 or sl_dist > 1.2 * atr:
                    sl = entry + 0.8 * atr
                    sl_dist = sl - entry

                if sl_dist > 0.2 * atr:
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
