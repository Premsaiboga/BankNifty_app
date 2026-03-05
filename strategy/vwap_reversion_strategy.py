"""
VWAP Mean Reversion Strategy
==============================
Trades reversions back to VWAP when price deviates significantly.

Logic:
- BUY: Price drops below VWAP - 0.7*ATR, then shows strong bullish reversal
- SELL: Price rises above VWAP + 0.7*ATR, then shows strong bearish reversal
- SL: Minimum 1.0 ATR from entry
- Target: Back toward VWAP (RR 1.5)
"""

import pandas as pd
from ml.features import extract_features


class VWAPReversionStrategy:
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
            if pd.isna(curr.get("vwap")):
                continue

            atr = curr["atr"]
            vwap = curr["vwap"]
            close = curr["close"]

            # ANTI-CHOP FILTER: if price is whipsawing around VWAP (tiny deviations),
            # the "reversion" isn't real — it's just noise
            range_vs_avg = curr.get("range_vs_avg", 1.0)

            # ===== BUY: Price dropped well below VWAP, reverting up =====
            deviation_down = vwap - close
            if (
                deviation_down > 0.7 * atr   # Significant deviation
                and close > curr["open"]       # Bullish reversal candle
                and curr["body_ratio"] > 0.45  # Strong body
                and prev["close"] < vwap       # Prev also below VWAP
                and curr["rsi_14"] < 45        # Actually oversold
                and range_vs_avg > 0.8         # Reversal candle has some substance
            ):
                entry = close
                sl = min(curr["low"], prev["low"]) - 0.1 * atr
                sl_dist = entry - sl

                # Enforce minimum 1.0 ATR SL
                if sl_dist < 1.0 * atr:
                    sl = entry - 1.0 * atr
                    sl_dist = entry - sl

                # Cap at 2.0 ATR max
                if sl_dist > 2.0 * atr:
                    continue

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

            # ===== SELL: Price rose well above VWAP, reverting down =====
            deviation_up = close - vwap
            if (
                deviation_up > 0.7 * atr      # Significant deviation
                and close < curr["open"]        # Bearish reversal candle
                and curr["body_ratio"] > 0.45   # Strong body
                and prev["close"] > vwap        # Prev also above VWAP
                and curr["rsi_14"] > 55         # Actually overbought
                and range_vs_avg > 0.8          # Reversal candle has some substance
            ):
                entry = close
                sl = max(curr["high"], prev["high"]) + 0.1 * atr
                sl_dist = sl - entry

                # Enforce minimum 1.0 ATR SL
                if sl_dist < 1.0 * atr:
                    sl = entry + 1.0 * atr
                    sl_dist = sl - entry

                if sl_dist > 2.0 * atr:
                    continue

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
