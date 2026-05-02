"""
Build Training Data V3
======================
Generates labeled training data using the same fixed-target plus protective
trailing exit plan used by live trading and backtests.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from strategy.indicators import calculate_all_indicators
from strategy.orb_strategy import ORBStrategy
from strategy.ema_scalp_strategy import EMAScalpStrategy
from strategy.momentum_surge_strategy import MomentumSurgeStrategy
from strategy.vwap_reversion_strategy import VWAPReversionStrategy
from strategy.pivot_scalp_strategy import PivotScalpStrategy
from ml.features import FEATURE_COLUMNS
from config import (
    EXIT_TARGET_R,
    TRAIL_BREAKEVEN_R,
    TRAIL_FIXED_TARGET_SL,
    TRAIL_LOCK_PROFIT_R,
    TRAIL_LOCK_TRIGGER_R,
    USE_FIXED_TARGET_EXIT,
)

# =========================
# CONFIG
# =========================
DATA_PATH = PROJECT_ROOT / "data/historical/banknifty_5m.csv"
OUTPUT_PATH = PROJECT_ROOT / "ml/training_data_v2.csv"
SLIPPAGE_POINTS = 5

# =========================
# LOAD DATA
# =========================
print("Loading historical data...")
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

print(f"Loaded {len(df)} candles from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

# =========================
# CALCULATE ALL INDICATORS
# =========================
print("Calculating indicators...")
df = calculate_all_indicators(df)

df = df.dropna(subset=["atr", "rsi_14", "ema_9", "ema_21", "bb_upper"]).reset_index(drop=True)
print(f"After indicator warmup: {len(df)} candles")

# =========================
# TRADE EVALUATOR WITH LIVE EXIT PLAN
# =========================
def evaluate_trade(df: pd.DataFrame, trade: dict, max_candles: int = 60) -> int:
    """
    Evaluate trade with the configured live exit plan.
    Returns: 1 = profitable exit (target/trail/breakeven/EOD+), 0 = loss
    """
    entry = trade["entry"]
    sl = trade["stoploss"]
    trade_type = trade["type"]
    entry_time = trade["time"]

    # Apply slippage
    if trade_type == "BUY":
        entry += SLIPPAGE_POINTS
    else:
        entry -= SLIPPAGE_POINTS

    idx_list = df.index[df["datetime"] == entry_time].tolist()
    if not idx_list:
        return 0

    idx = idx_list[0]
    end_idx = min(idx + max_candles, len(df))
    entry_date = df.iloc[idx]["date"]

    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return 0

    current_sl = sl
    if USE_FIXED_TARGET_EXIT:
        if trade_type == "BUY":
            target = entry + EXIT_TARGET_R * sl_dist
        else:
            target = entry - EXIT_TARGET_R * sl_dist

        for i in range(idx + 1, end_idx):
            candle = df.iloc[i]

            if trade_type == "BUY":
                if candle["date"] != entry_date:
                    return 1 if df.iloc[i - 1]["close"] > entry else 0
                if candle["low"] <= current_sl:
                    return 1 if current_sl > entry else 0
                if candle["high"] >= target:
                    return 1
            else:
                if candle["date"] != entry_date:
                    return 1 if df.iloc[i - 1]["close"] < entry else 0
                if candle["high"] >= current_sl:
                    return 1 if current_sl < entry else 0
                if candle["low"] <= target:
                    return 1

            if TRAIL_FIXED_TARGET_SL:
                if trade_type == "BUY":
                    favorable_r = (candle["high"] - entry) / sl_dist
                    if favorable_r >= TRAIL_BREAKEVEN_R:
                        current_sl = max(current_sl, entry + 2)
                    if favorable_r >= TRAIL_LOCK_TRIGGER_R:
                        current_sl = max(current_sl, entry + TRAIL_LOCK_PROFIT_R * sl_dist)
                else:
                    favorable_r = (entry - candle["low"]) / sl_dist
                    if favorable_r >= TRAIL_BREAKEVEN_R:
                        current_sl = min(current_sl, entry - 2)
                    if favorable_r >= TRAIL_LOCK_TRIGGER_R:
                        current_sl = min(current_sl, entry - TRAIL_LOCK_PROFIT_R * sl_dist)

        return 0

    best_move = 0
    for i in range(idx + 1, end_idx):
        candle = df.iloc[i]

        if candle["date"] != entry_date:
            prev = df.iloc[i - 1]
            if trade_type == "BUY":
                return 1 if prev["close"] > entry else 0
            else:
                return 1 if prev["close"] < entry else 0

        if trade_type == "BUY":
            move_r = (candle["high"] - entry) / sl_dist
        else:
            move_r = (entry - candle["low"]) / sl_dist
        best_move = max(best_move, move_r)

        if best_move >= TRAIL_BREAKEVEN_R:
            current_sl = max(current_sl, entry + 2) if trade_type == "BUY" else min(current_sl, entry - 2)

        if best_move >= 1.5:
            current_sl = (
                max(current_sl, entry + (best_move - 0.5) * sl_dist)
                if trade_type == "BUY"
                else min(current_sl, entry - (best_move - 0.5) * sl_dist)
            )

        # Check SL (no fixed target check — trailing handles exits)
        if trade_type == "BUY":
            if candle["low"] <= current_sl:
                pnl = current_sl - entry
                return 1 if pnl > 0 else 0
        else:
            if candle["high"] >= current_sl:
                pnl = entry - current_sl
                return 1 if pnl > 0 else 0

    return 0  # Timeout = loss


# =========================
# INIT STRATEGIES
# =========================
strategies = [
    ORBStrategy(rr=2.0),
    EMAScalpStrategy(rr=2.0),
    MomentumSurgeStrategy(rr=2.0),
    VWAPReversionStrategy(rr=2.0),
    PivotScalpStrategy(rr=2.0),
]

# =========================
# GENERATE TRADES
# =========================
all_rows = []
strategy_counts = {}

for strat in strategies:
    name = strat.__class__.__name__
    print(f"\nGenerating trades for {name}...")

    trades = strat.generate_trades(df)
    print(f"  Raw signals: {len(trades)}")

    wins = 0
    losses = 0

    for trade in trades:
        result = evaluate_trade(df, trade)

        if result == 1:
            wins += 1
        else:
            losses += 1

        row = {**trade["features"]}
        row["result"] = result
        row["strategy_name"] = trade["strategy"]
        row["type"] = trade["type"]
        row["entry"] = trade["entry"]
        row["stoploss"] = trade["stoploss"]
        row["target"] = trade["target"]
        row["rr"] = trade["rr"]
        row["time"] = trade["time"]

        all_rows.append(row)

    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    strategy_counts[trade["strategy"]] = {"total": total, "wins": wins, "wr": wr}
    print(f"  Results (live exit plan): {wins}W / {losses}L = {wr:.1f}% win rate")

# =========================
# SAVE CSV
# =========================
df_out = pd.DataFrame(all_rows)
df_out.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*50}")
print(f"TRAINING DATA V3 SUMMARY (live exit plan)")
print(f"{'='*50}")
print(f"Total samples: {len(df_out)}")
print(f"Win rate: {df_out['result'].mean()*100:.1f}%")
print(f"\nBy strategy:")
for strat, stats in strategy_counts.items():
    print(f"  {strat}: {stats['total']} trades, {stats['wr']:.1f}% win rate")
print(f"\nSaved to: {OUTPUT_PATH}")
