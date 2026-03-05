"""
Comprehensive Backtest V3 — Dynamic Trailing Stops
====================================================
Backtests all 5 strategies with AI filtering + DYNAMIC TRAILING.
NO fixed target — trailing stops let winners run to 1:2, 1:3, 1:4+.

Exit Logic (per candle):
1. Check trailed SL hit → exit at trail level
2. If price moved 0.8R in favor → move SL to BREAKEVEN
3. If price moved 1.5R+ → trail SL dynamically (locks move - 0.5R)
4. End of day → exit at close price
5. NO fixed target cap — trailing captures maximum move (1:2, 1:3, 1:4+)
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
CAPITAL = 10000
RISK_PER_TRADE = 750
LOT_SIZE = 15
SLIPPAGE_POINTS = 5

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
# TRADE EVALUATOR WITH TRAILING STOP
# =========================
def evaluate_trade(df, trade, max_candles=60):
    """
    Evaluate trade with DYNAMIC TRAILING STOP logic:
    - Move SL to breakeven after 0.8R profit
    - Trail SL at (move - 0.5R) after 1.5R profit
    - NO fixed target cap — lets winners run to 1:2, 1:3, 1:4+
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
        return {"result": "SKIP", "pnl_r": 0}

    idx = idx_list[0]
    entry_date = df.iloc[idx]["date"]
    end_idx = min(idx + max_candles, len(df))

    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return {"result": "SKIP", "pnl_r": 0}

    current_sl = sl
    best_move = 0  # Best favorable move in R units
    trailed = False

    for i in range(idx + 1, end_idx):
        candle = df.iloc[i]

        # End of day exit
        if candle["date"] != entry_date:
            prev = df.iloc[i - 1]
            if trade_type == "BUY":
                pnl_r = (prev["close"] - entry) / sl_dist
            else:
                pnl_r = (entry - prev["close"]) / sl_dist
            result = "EOD_WIN" if pnl_r > 0 else "EOD_LOSS"
            return {"result": result, "pnl_r": round(pnl_r, 2)}

        # Calculate current favorable move
        if trade_type == "BUY":
            move = candle["high"] - entry  # Best move this candle
            move_r = move / sl_dist
        else:
            move = entry - candle["low"]
            move_r = move / sl_dist

        best_move = max(best_move, move_r)

        # === DYNAMIC TRAILING STOP LOGIC ===
        # Breakeven at 0.8R (gives trade room to breathe)
        # Trail at 1.5R+ (locks move - 0.5R, so at 2R locks 1.5R, at 3R locks 2.5R)
        # NO fixed target — let winners run to 1:2, 1:3, 1:4+

        # After 0.8R profit: move SL to breakeven
        if best_move >= 0.8 and not trailed:
            if trade_type == "BUY":
                new_sl = entry + 2  # Small buffer above entry
                current_sl = max(current_sl, new_sl)
            else:
                new_sl = entry - 2
                current_sl = min(current_sl, new_sl)
            trailed = True

        # After 1.5R profit: trail SL dynamically (locks move - 0.5R)
        # At 2R move → SL at 1.5R profit, at 3R → SL at 2.5R, at 4R → SL at 3.5R
        if best_move >= 1.5:
            if trade_type == "BUY":
                trail_sl = entry + (best_move - 0.5) * sl_dist
                current_sl = max(current_sl, trail_sl)
            else:
                trail_sl = entry - (best_move - 0.5) * sl_dist
                current_sl = min(current_sl, trail_sl)

        # Check SL hit (use current_sl which may be trailed)
        if trade_type == "BUY":
            if candle["low"] <= current_sl:
                pnl_r = (current_sl - entry) / sl_dist
                if pnl_r > 0:
                    return {"result": "TRAIL_WIN", "pnl_r": round(pnl_r, 2)}
                elif pnl_r >= -0.1:  # Breakeven (small tolerance)
                    return {"result": "BREAKEVEN", "pnl_r": 0}
                else:
                    return {"result": "LOSS", "pnl_r": -1}

        else:  # SELL
            if candle["high"] >= current_sl:
                pnl_r = (entry - current_sl) / sl_dist
                if pnl_r > 0:
                    return {"result": "TRAIL_WIN", "pnl_r": round(pnl_r, 2)}
                elif pnl_r >= -0.1:
                    return {"result": "BREAKEVEN", "pnl_r": 0}
                else:
                    return {"result": "LOSS", "pnl_r": -1}

    # Timeout — use current position
    last = df.iloc[end_idx - 1]
    if trade_type == "BUY":
        pnl_r = (last["close"] - entry) / sl_dist
    else:
        pnl_r = (entry - last["close"]) / sl_dist
    return {"result": "TIMEOUT", "pnl_r": round(pnl_r, 2)}

# =========================
# STRATEGIES
# =========================
strategies = [
    ORBStrategy(rr=2.0),
    EMAScalpStrategy(rr=2.0),
    VWAPReversionStrategy(rr=2.0),
    MomentumSurgeStrategy(rr=2.0),
    PivotScalpStrategy(rr=2.0),
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
print("\nApplying AI filter and evaluating with trailing stops...")

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

    # Evaluate trade WITH trailing stops
    eval_result = evaluate_trade(df, trade)

    if eval_result["result"] == "SKIP":
        continue

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
print(f"  BACKTEST RESULTS V3 (with trailing stops)")
print(f"{'='*60}")

total_trades = len(df_results)
trail_wins = len(df_results[df_results["result"] == "TRAIL_WIN"])
breakevens = len(df_results[df_results["result"] == "BREAKEVEN"])
eod_wins = len(df_results[df_results["result"] == "EOD_WIN"])
eod_losses = len(df_results[df_results["result"] == "EOD_LOSS"])
losses = len(df_results[df_results["result"] == "LOSS"])
timeouts = len(df_results[df_results["result"] == "TIMEOUT"])

# Profitable = TRAIL_WIN + EOD_WIN + BREAKEVEN
profitable = trail_wins + eod_wins + breakevens
profit_rate = profitable / total_trades * 100 if total_trades > 0 else 0
net_pnl_r = df_results["pnl_r"].sum()

# RR distribution of trail wins
tw = df_results[df_results["result"] == "TRAIL_WIN"]
r1_wins = len(tw[tw["pnl_r"] < 1.5])  # 1:1 range
r2_wins = len(tw[(tw["pnl_r"] >= 1.5) & (tw["pnl_r"] < 2.5)])  # 1:2 range
r3_wins = len(tw[(tw["pnl_r"] >= 2.5) & (tw["pnl_r"] < 3.5)])  # 1:3 range
r4_plus = len(tw[tw["pnl_r"] >= 3.5])  # 1:4+ range

print(f"\nTotal AI-filtered trades: {total_trades}")
print(f"\n  Trail Stop Wins:           {trail_wins:>4} trades  (+{tw['pnl_r'].sum():.1f}R)")
print(f"    ├─ 1:1 range (<1.5R):    {r1_wins:>4} trades")
print(f"    ├─ 1:2 range (1.5-2.5R): {r2_wins:>4} trades")
print(f"    ├─ 1:3 range (2.5-3.5R): {r3_wins:>4} trades")
print(f"    └─ 1:4+ range (>3.5R):   {r4_plus:>4} trades")
print(f"  Breakeven Exits:           {breakevens:>4} trades  ( 0R)")
print(f"  EOD Wins:                  {eod_wins:>4} trades  (+{df_results[df_results['result']=='EOD_WIN']['pnl_r'].sum():.1f}R)")
print(f"  EOD Losses:                {eod_losses:>4} trades  ({df_results[df_results['result']=='EOD_LOSS']['pnl_r'].sum():.1f}R)")
print(f"  Full SL Losses:            {losses:>4} trades  (-{losses:.0f}R)")
print(f"  Timeouts:                  {timeouts:>4} trades")

print(f"\n  PROFITABLE EXITS: {profitable}/{total_trades} = {profit_rate:.1f}%")
print(f"  NET PnL: {net_pnl_r:.1f}R")

# Avg risk per trade
avg_sl_points = (df_results["entry"] - df_results["stoploss"]).abs().mean()
approx_option_sl = avg_sl_points * 0.4
profit_per_r = approx_option_sl * LOT_SIZE

print(f"\nAvg SL distance: {avg_sl_points:.0f} BankNifty points")
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
    s_profitable = len(subset[subset["pnl_r"] > 0]) + len(subset[subset["result"] == "BREAKEVEN"])
    s_total = len(subset)
    s_wr = s_profitable / s_total * 100 if s_total > 0 else 0
    s_pnl = subset["pnl_r"].sum()

    print(f"\n{strat_name}:")
    print(f"  Trades: {s_total} | Profitable: {s_profitable} | Rate: {s_wr:.1f}% | PnL: {s_pnl:.1f}R")

# =========================
# MONTHLY BREAKDOWN
# =========================
print(f"\n{'='*60}")
print(f"  MONTHLY BREAKDOWN")
print(f"{'='*60}")

monthly = df_results.groupby("month").agg(
    trades=("result", "count"),
    profitable=("pnl_r", lambda x: (x > 0).sum() + (x == 0).sum()),
    losses=("pnl_r", lambda x: (x < 0).sum()),
    net_pnl=("pnl_r", "sum"),
).reset_index()

monthly["profit_rate"] = (monthly["profitable"] / monthly["trades"] * 100).round(1)
monthly["trades_per_day"] = (monthly["trades"] / 22).round(1)

print(f"\n{'Month':<10} {'Trades':>7} {'Prof':>5} {'Loss':>5} {'Rate%':>6} {'PnL(R)':>8} {'Tr/Day':>7}")
print("-" * 55)

for _, row in monthly.iterrows():
    print(f"{row['month']:<10} {row['trades']:>7} {row['profitable']:>5} {row['losses']:>5} "
          f"{row['profit_rate']:>5.1f}% {row['net_pnl']:>7.1f}R {row['trades_per_day']:>6.1f}")

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

for conf in ["HIGH", "MEDIUM", "LOW", "NO_MODEL"]:
    subset = df_results[df_results["ai_confidence"] == conf]
    if len(subset) == 0:
        continue
    c_profitable = len(subset[subset["pnl_r"] > 0]) + len(subset[subset["result"] == "BREAKEVEN"])
    c_rate = c_profitable / len(subset) * 100
    c_pnl = subset["pnl_r"].sum()
    print(f"{conf}: {len(subset)} trades | Profitable: {c_rate:.1f}% | PnL: {c_pnl:.1f}R")

# =========================
# SAVE DETAILED RESULTS
# =========================
output_path = PROJECT_ROOT / "backtest/backtest_results_v2.csv"
df_results.to_csv(output_path, index=False)
print(f"\nDetailed results saved to: {output_path}")
