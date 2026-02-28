"""
Build Training Data V2
======================
Generates labeled training data from all 5 strategies on historical data.
Uses comprehensive feature engineering (22 features).
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

# Drop rows with NaN indicators (first ~20 candles)
df = df.dropna(subset=["atr", "rsi_14", "ema_9", "ema_21", "bb_upper"]).reset_index(drop=True)
print(f"After indicator warmup: {len(df)} candles")

# =========================
# TRADE EVALUATOR
# =========================
def evaluate_trade(df: pd.DataFrame, trade: dict, max_candles: int = 60) -> int:
    """
    Evaluate if trade hits target or SL first.
    max_candles: Max forward look (60 x 5min = 5 hours)
    Returns: 1=WIN, 0=LOSS
    """
    entry = trade["entry"]
    sl = trade["stoploss"]
    target = trade["target"]
    trade_type = trade["type"]
    entry_time = trade["time"]

    idx_list = df.index[df["datetime"] == entry_time].tolist()
    if not idx_list:
        return 0

    idx = idx_list[0]
    end_idx = min(idx + max_candles, len(df))

    for i in range(idx + 1, end_idx):
        candle = df.iloc[i]

        # Check for new day (close open positions at end of day)
        if candle["date"] != df.iloc[idx]["date"]:
            # Check if we'd be profitable at day end
            prev_candle = df.iloc[i - 1]
            if trade_type == "BUY":
                return 1 if prev_candle["close"] > entry else 0
            else:
                return 1 if prev_candle["close"] < entry else 0

        if trade_type == "BUY":
            if candle["low"] <= sl:
                return 0  # SL hit
            if candle["high"] >= target:
                return 1  # Target hit
        else:
            if candle["high"] >= sl:
                return 0  # SL hit
            if candle["low"] <= target:
                return 1  # Target hit

    return 0  # Didn't reach either → loss


# =========================
# INIT STRATEGIES
# =========================
strategies = [
    ORBStrategy(rr=2.0),
    EMAScalpStrategy(rr=1.5),
    VWAPReversionStrategy(rr=1.5),
    MomentumSurgeStrategy(rr=2.0),
    PivotScalpStrategy(rr=1.5),
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
    print(f"  Results: {wins}W / {losses}L = {wr:.1f}% win rate")

# =========================
# SAVE CSV
# =========================
df_out = pd.DataFrame(all_rows)
df_out.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*50}")
print(f"TRAINING DATA V2 SUMMARY")
print(f"{'='*50}")
print(f"Total samples: {len(df_out)}")
print(f"Win rate: {df_out['result'].mean()*100:.1f}%")
print(f"\nBy strategy:")
for strat, stats in strategy_counts.items():
    print(f"  {strat}: {stats['total']} trades, {stats['wr']:.1f}% win rate")
print(f"\nSaved to: {OUTPUT_PATH}")
