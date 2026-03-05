"""
Build Training Data V3
======================
Generates labeled training data with TRAILING STOP evaluation.
A trade is labeled WIN (1) if it exits with any profit (target hit,
trail win, or EOD win). This teaches the ML model to select trades
that will be profitable under our trailing stop exit system.
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
from strategy.vwap_reversion_strategy import VWAPReversionStrategy
from strategy.momentum_surge_strategy import MomentumSurgeStrategy
from strategy.pivot_scalp_strategy import PivotScalpStrategy
from ml.features import FEATURE_COLUMNS

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
# TRADE EVALUATOR WITH TRAILING STOPS
# =========================
def evaluate_trade(df: pd.DataFrame, trade: dict, max_candles: int = 60) -> int:
    """
    Evaluate trade with trailing stop logic.
    Returns: 1 = profitable exit (target/trail/breakeven/EOD+), 0 = loss
    """
    entry = trade["entry"]
    sl = trade["stoploss"]
    target = trade["target"]
    trade_type = trade["type"]
    entry_time = trade["time"]

    # Apply slippage
    if trade_type == "BUY":
        entry += SLIPPAGE_POINTS
        target += SLIPPAGE_POINTS
    else:
        entry -= SLIPPAGE_POINTS
        target -= SLIPPAGE_POINTS

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
    best_move = 0

    for i in range(idx + 1, end_idx):
        candle = df.iloc[i]

        # End of day
        if candle["date"] != entry_date:
            prev = df.iloc[i - 1]
            if trade_type == "BUY":
                return 1 if prev["close"] > entry else 0
            else:
                return 1 if prev["close"] < entry else 0

        # Calculate favorable move
        if trade_type == "BUY":
            move_r = (candle["high"] - entry) / sl_dist
        else:
            move_r = (entry - candle["low"]) / sl_dist

        best_move = max(best_move, move_r)

        # Trail: breakeven at 0.8R, dynamic trail at 1.5R+
        # NO fixed target — let trailing capture 1:2, 1:3, 1:4+
        if best_move >= 0.8:
            if trade_type == "BUY":
                new_sl = entry + 2
                current_sl = max(current_sl, new_sl)
            else:
                new_sl = entry - 2
                current_sl = min(current_sl, new_sl)

        if best_move >= 1.5:
            if trade_type == "BUY":
                trail_sl = entry + (best_move - 0.5) * sl_dist
                current_sl = max(current_sl, trail_sl)
            else:
                trail_sl = entry - (best_move - 0.5) * sl_dist
                current_sl = min(current_sl, trail_sl)

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
    VWAPReversionStrategy(rr=2.0),
    MomentumSurgeStrategy(rr=2.0),
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
    print(f"  Results (with trailing): {wins}W / {losses}L = {wr:.1f}% win rate")

# =========================
# SAVE CSV
# =========================
df_out = pd.DataFrame(all_rows)
df_out.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*50}")
print(f"TRAINING DATA V3 SUMMARY (with trailing stops)")
print(f"{'='*50}")
print(f"Total samples: {len(df_out)}")
print(f"Win rate: {df_out['result'].mean()*100:.1f}%")
print(f"\nBy strategy:")
for strat, stats in strategy_counts.items():
    print(f"  {strat}: {stats['total']} trades, {stats['wr']:.1f}% win rate")
print(f"\nSaved to: {OUTPUT_PATH}")
