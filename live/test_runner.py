import sys
from pathlib import Path
import pandas as pd

# =========================
# PATH FIX
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# =========================
# IMPORTS
# =========================
from strategy.vwap_strategy import VWAPPullbackStrategy
from strategy.pivot_strategy import PivotStrategy
from strategy.abcd_strategy import ABCDStrategy

from brain.market_brain import detect_bias, detect_regime
from brain.strategy_brain import allow_trade
from brain.risk_brain import adjust_targets
from brain.liquidity_engine import liquidity_block
from ml.ai_filter import ai_filter

# =========================
# LOAD DATA
# =========================
DATA = PROJECT_ROOT / "data/historical/banknifty_5m.csv"

df = pd.read_csv(DATA)
df["datetime"] = pd.to_datetime(df["datetime"])

# =========================
# BUILD REQUIRED INDICATORS
# =========================
df["tp"] = (df["high"] + df["low"] + df["close"]) / 3

df["vwap"] = (
    df.groupby(df["datetime"].dt.date)["tp"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift()).abs()
low_close = (df["low"] - df["close"].shift()).abs()

df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()

df["date"] = df["datetime"].dt.date

daily = df.groupby("date").agg(
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
)

daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3
daily["r1"] = 2 * daily["pivot"] - daily["low"]
daily["s1"] = 2 * daily["pivot"] - daily["high"]

df = df.merge(
    daily[["pivot", "r1", "s1"]],
    left_on="date",
    right_index=True,
    how="left",
)

# =========================
# INIT STRATEGIES
# =========================
vwap = VWAPPullbackStrategy()
pivot = PivotStrategy()
abcd = ABCDStrategy()

print("\n▶ Running offline simulation...\n")

approved = 0
blocked_liquidity = 0
ai_rejected = 0

# =========================
# SIMULATION LOOP
# =========================
for i in range(200, len(df), 5):

    sub = df.iloc[:i].copy()

    bias, _ = detect_bias(sub)
    regime = detect_regime(sub)

    trades = []

    # --- ABCD ---
    for t in abcd.generate_trades(sub):
        t["strategy"] = "ABCD"
        trades.append(t)

    # --- PIVOT ---
    for t in pivot.generate_trades(sub):
        t["strategy"] = "PIVOT"
        trades.append(t)

    # --- VWAP ---
    for t in vwap.generate_signals(sub):
        t["strategy"] = "VWAP_PULLBACK"
        trades.append(t)

    # =========================
    # TRADE PIPELINE
    # =========================
    for trade in trades:

        if not allow_trade(trade, bias, regime):
            continue

        if liquidity_block(trade, sub):
            blocked_liquidity += 1
            continue

        trade = adjust_targets(trade, sub)

        last = sub.iloc[-1]

        # features required by model
        trade["features"] = {
            "vwap_distance": abs(trade["entry"] - last["vwap"]),
            "candle_size": abs(trade["entry"] - trade["stoploss"]),
            "atr": last["atr"],
            "pattern_strength": trade.get("pattern_strength", 0),
        }

        ai = ai_filter(trade, sub, bias, regime)

        if ai["decision"] == "SKIP":
            ai_rejected += 1
            continue

        approved += 1

        print(
            f"✅ {trade['strategy']} {trade['type']} | "
            f"Prob={ai['probability']} | "
            f"Bias={bias} | Regime={regime}"
        )

# =========================
# SUMMARY
# =========================
print("\n==============================")
print("✅ Simulation completed")
print(f"Approved Trades : {approved}")
print(f"Liquidity Blocks: {blocked_liquidity}")
print(f"AI Rejections   : {ai_rejected}")
print("==============================")