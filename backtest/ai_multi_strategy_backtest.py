import sys
from pathlib import Path

# =========================
# ADD PROJECT ROOT TO PATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from strategy.vwap_strategy import VWAPPullbackStrategy
from strategy.pivot_strategy import PivotStrategy
from strategy.abcd_strategy import ABCDStrategy
from ml.ai_filter import ai_filter
# =========================
# CONFIG
# =========================
DATA_PATH = Path("../data/historical/banknifty_5m.csv")
RR = 4

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

# =========================
# INIT STRATEGIES
# =========================
vwap = VWAPPullbackStrategy(rr=RR)
pivot = PivotStrategy(rr=RR)
abcd = ABCDStrategy(rr=RR)

# =========================
# COLLECT TRADES
# =========================
trades = []
trades += vwap.generate_trades(df)
trades += pivot.generate_trades(df)
trades += abcd.generate_trades(df)

print("Total raw trades:", len(trades))

# =========================
# BACKTEST WITH AI
# =========================
results = []

for trade in trades:
    decision = ai_filter(trade)

    if decision["decision"] == "SKIP":
        continue

    entry_time = trade["time"]
    idx = df.index[df["datetime"] == entry_time][0]

    entry = trade["entry"]
    sl = trade["stoploss"]
    rr = trade["rr"]

    if trade["type"] == "BUY":
        target = entry + (entry - sl) * rr
    else:
        target = entry - (sl - entry) * rr

    result = "BE"

    for i in range(idx + 1, len(df)):
        candle = df.iloc[i]

        if trade["type"] == "BUY":
            if candle["low"] <= sl:
                result = "LOSS"
                break
            if candle["high"] >= target:
                result = "WIN"
                break
        else:
            if candle["high"] >= sl:
                result = "LOSS"
                break
            if candle["low"] <= target:
                result = "WIN"
                break

    results.append({
        "time": trade["time"],
        "strategy": trade["strategy"],
        "result": result,
        "probability": decision["probability"]
    })

# =========================
# RESULTS SUMMARY
# =========================
res_df = pd.DataFrame(results)
res_df["month"] = pd.to_datetime(res_df["time"]).dt.to_period("M")

monthly = res_df.groupby("month").agg(
    trades=("result", "count"),
    wins=("result", lambda x: (x == "WIN").sum()),
    losses=("result", lambda x: (x == "LOSS").sum())
)

monthly["net_pnl_R"] = monthly["wins"] * RR - monthly["losses"]

print("\n📊 AI FILTERED MONTHLY RESULTS\n")
print(monthly)

print("\n📈 OVERALL AI STATS")
print("Total AI trades :", len(res_df))
print("Win rate        :", (res_df["result"] == "WIN").mean())
print("Net PnL (R)     :", monthly["net_pnl_R"].sum())

print("\n📌 Strategy-wise trade count")
print(res_df["strategy"].value_counts())
