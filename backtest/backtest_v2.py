"""
Comprehensive Backtest V2
==========================
Backtests all 5 strategies with AI filtering on historical data.
Provides monthly breakdown, per-strategy stats, and profit projections.
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
from ml.ai_filter_v2 import ai_filter_v2

# =========================
# CONFIG
# =========================
DATA_PATH = PROJECT_ROOT / "data/historical/banknifty_5m.csv"
CAPITAL = 10000  # Starting capital
RISK_PER_TRADE = 750  # Max risk per trade in Rs (option premium SL)
LOT_SIZE = 15

# =========================
# LOAD DATA
# =========================
print("Loading historical data...")
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
print(f"Loaded {len(df)} candles")

# =========================
# CALCULATE INDICATORS
# =========================
print("Calculating indicators...")
df = calculate_all_indicators(df)
df = df.dropna(subset=["atr", "rsi_14", "ema_9", "ema_21", "bb_upper"]).reset_index(drop=True)
print(f"After warmup: {len(df)} candles")

# =========================
# TRADE EVALUATOR
# =========================
def evaluate_trade(df, trade, max_candles=60):
    entry = trade["entry"]
    sl = trade["stoploss"]
    target = trade["target"]
    trade_type = trade["type"]
    entry_time = trade["time"]
    rr = trade["rr"]

    idx_list = df.index[df["datetime"] == entry_time].tolist()
    if not idx_list:
        return {"result": "SKIP", "pnl_r": 0}

    idx = idx_list[0]
    entry_date = df.iloc[idx]["date"]
    end_idx = min(idx + max_candles, len(df))

    for i in range(idx + 1, end_idx):
        candle = df.iloc[i]

        # End of day exit
        if candle["date"] != entry_date:
            prev = df.iloc[i - 1]
            if trade_type == "BUY":
                pnl = (prev["close"] - entry) / (entry - sl) if entry != sl else 0
            else:
                pnl = (entry - prev["close"]) / (sl - entry) if sl != entry else 0
            return {"result": "EOD", "pnl_r": round(pnl, 2)}

        if trade_type == "BUY":
            if candle["low"] <= sl:
                return {"result": "LOSS", "pnl_r": -1}
            if candle["high"] >= target:
                return {"result": "WIN", "pnl_r": rr}
        else:
            if candle["high"] >= sl:
                return {"result": "LOSS", "pnl_r": -1}
            if candle["low"] <= target:
                return {"result": "WIN", "pnl_r": rr}

    return {"result": "TIMEOUT", "pnl_r": 0}

# =========================
# STRATEGIES
# =========================
strategies = [
    ORBStrategy(rr=2.0),
    EMAScalpStrategy(rr=1.5),
    VWAPReversionStrategy(rr=1.5),
    MomentumSurgeStrategy(rr=2.0),
    PivotScalpStrategy(rr=1.5),
]

# =========================
# RUN BACKTEST
# =========================
print("\nGenerating trades from all strategies...")

all_trades = []
for strat in strategies:
    name = strat.__class__.__name__
    trades = strat.generate_trades(df)
    print(f"  {name}: {len(trades)} raw signals")
    all_trades.extend(trades)

# Sort by time
all_trades.sort(key=lambda t: t["time"])
print(f"\nTotal raw signals: {len(all_trades)}")

# =========================
# AI FILTER + EVALUATE
# =========================
print("\nApplying AI filter and evaluating...")

results = []
daily_trades = {}

for trade in all_trades:
    date = trade["time"].date() if hasattr(trade["time"], "date") else pd.to_datetime(trade["time"]).date()
    daily_trades.setdefault(date, 0)

    # Max 5 trades per day
    if daily_trades[date] >= 5:
        continue

    # AI Filter
    ai_result = ai_filter_v2(trade)

    if ai_result["decision"] != "TAKE":
        continue

    # Evaluate trade
    eval_result = evaluate_trade(df, trade)

    daily_trades[date] += 1

    results.append({
        "date": date,
        "month": str(date)[:7],
        "strategy": trade["strategy"],
        "type": trade["type"],
        "entry": trade["entry"],
        "stoploss": trade["stoploss"],
        "target": trade["target"],
        "rr": trade["rr"],
        "ai_prob": ai_result["probability"],
        "ai_confidence": ai_result["confidence"],
        "result": eval_result["result"],
        "pnl_r": eval_result["pnl_r"],
    })

df_results = pd.DataFrame(results)

if len(df_results) == 0:
    print("\nNo trades passed AI filter! Try lowering thresholds.")
    sys.exit()

# =========================
# RESULTS ANALYSIS
# =========================
print(f"\n{'='*60}")
print(f"  BACKTEST RESULTS V2")
print(f"{'='*60}")

total_trades = len(df_results)
wins = len(df_results[df_results["result"] == "WIN"])
losses = len(df_results[df_results["result"] == "LOSS"])
eod = len(df_results[df_results["result"] == "EOD"])
timeouts = len(df_results[df_results["result"] == "TIMEOUT"])

win_rate = wins / total_trades * 100 if total_trades > 0 else 0
net_pnl_r = df_results["pnl_r"].sum()

print(f"\nTotal AI-filtered trades: {total_trades}")
print(f"Wins: {wins} | Losses: {losses} | EOD: {eod} | Timeout: {timeouts}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Net PnL (R units): {net_pnl_r:.1f}R")

# Avg risk per trade (approximate)
avg_sl_points = (df_results["entry"] - df_results["stoploss"]).abs().mean()
approx_option_sl = avg_sl_points * 0.4  # Approximate delta for option premium
profit_per_r = approx_option_sl * LOT_SIZE

print(f"\nAvg SL distance: {avg_sl_points:.0f} BankNifty points")
print(f"Approx premium SL: ~₹{approx_option_sl:.0f} per unit")
print(f"Approx ₹ per R unit: ~₹{profit_per_r:.0f} (1 lot)")
print(f"Estimated total P&L: ~₹{net_pnl_r * profit_per_r:.0f}")

# =========================
# PER-STRATEGY BREAKDOWN
# =========================
print(f"\n{'='*60}")
print(f"  PER-STRATEGY BREAKDOWN")
print(f"{'='*60}")

for strat_name in df_results["strategy"].unique():
    subset = df_results[df_results["strategy"] == strat_name]
    s_wins = len(subset[subset["result"] == "WIN"])
    s_total = len(subset)
    s_wr = s_wins / s_total * 100 if s_total > 0 else 0
    s_pnl = subset["pnl_r"].sum()

    print(f"\n{strat_name}:")
    print(f"  Trades: {s_total} | Wins: {s_wins} | WR: {s_wr:.1f}% | PnL: {s_pnl:.1f}R")

# =========================
# MONTHLY BREAKDOWN
# =========================
print(f"\n{'='*60}")
print(f"  MONTHLY BREAKDOWN")
print(f"{'='*60}")

monthly = df_results.groupby("month").agg(
    trades=("result", "count"),
    wins=("result", lambda x: (x == "WIN").sum()),
    losses=("result", lambda x: (x == "LOSS").sum()),
    net_pnl=("pnl_r", "sum"),
).reset_index()

monthly["wr"] = (monthly["wins"] / monthly["trades"] * 100).round(1)
monthly["trades_per_day"] = (monthly["trades"] / 22).round(1)  # ~22 trading days

print(f"\n{'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'WR%':>6} {'PnL(R)':>8} {'Trades/Day':>10}")
print("-" * 55)

for _, row in monthly.iterrows():
    print(f"{row['month']:<10} {row['trades']:>7} {row['wins']:>5} {row['losses']:>5} "
          f"{row['wr']:>5.1f}% {row['net_pnl']:>7.1f}R {row['trades_per_day']:>9.1f}")

total_months = len(monthly)
avg_monthly_pnl = net_pnl_r / total_months if total_months > 0 else 0
avg_daily_trades = total_trades / (total_months * 22) if total_months > 0 else 0

print(f"\n{'='*60}")
print(f"  PROFIT PROJECTIONS (10K Capital)")
print(f"{'='*60}")
print(f"Avg trades/day: {avg_daily_trades:.1f}")
print(f"Avg monthly PnL: {avg_monthly_pnl:.1f}R = ~₹{avg_monthly_pnl * profit_per_r:.0f}")
print(f"Avg daily PnL: {avg_monthly_pnl/22:.1f}R = ~₹{avg_monthly_pnl * profit_per_r / 22:.0f}")

# =========================
# BY AI CONFIDENCE
# =========================
print(f"\n{'='*60}")
print(f"  BY AI CONFIDENCE LEVEL")
print(f"{'='*60}")

for conf in ["HIGH", "MEDIUM", "LOW"]:
    subset = df_results[df_results["ai_confidence"] == conf]
    if len(subset) == 0:
        continue
    c_wr = len(subset[subset["result"] == "WIN"]) / len(subset) * 100
    c_pnl = subset["pnl_r"].sum()
    print(f"{conf}: {len(subset)} trades | WR: {c_wr:.1f}% | PnL: {c_pnl:.1f}R")

# =========================
# SAVE DETAILED RESULTS
# =========================
output_path = PROJECT_ROOT / "backtest/backtest_results_v2.csv"
df_results.to_csv(output_path, index=False)
print(f"\nDetailed results saved to: {output_path}")
