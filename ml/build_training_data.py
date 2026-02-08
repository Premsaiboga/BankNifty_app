import sys
from pathlib import Path
import pandas as pd

# =========================
# PATH SETUP
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from strategy.vwap_strategy import VWAPPullbackStrategy
from strategy.pivot_strategy import PivotStrategy
from strategy.abcd_strategy import ABCDStrategy

# =========================
# CONFIG
# =========================
DATA_PATH = PROJECT_ROOT / "data/historical/banknifty_5m.csv"
OUTPUT_PATH = PROJECT_ROOT / "ml/training_data.csv"
RR = 4

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

# =========================
# INDICATORS (GLOBAL)
# =========================
# VWAP (intraday proxy)
df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
df["vwap"] = (
    df.groupby(df["datetime"].dt.date)["tp"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

# ATR
high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift()).abs()
low_close = (df["low"] - df["close"].shift()).abs()
df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()

# =========================
# DAILY PIVOT POINTS
# =========================
df["date"] = df["datetime"].dt.date

daily = df.groupby("date").agg(
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last")
)

daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3
daily["r1"] = 2 * daily["pivot"] - daily["low"]
daily["s1"] = 2 * daily["pivot"] - daily["high"]

# map back to intraday candles
df = df.merge(
    daily[["pivot", "r1", "s1"]],
    left_on="date",
    right_index=True,
    how="left"
)


# =========================
# TRADE EVALUATOR
# =========================
def evaluate_trade(df, trade):
    """
    Returns:
    1 = WIN
    0 = LOSS (includes BE)
    """
    entry = trade["entry"]
    sl = trade["stoploss"]
    rr = trade["rr"]
    trade_type = trade["type"]
    entry_time = trade["time"]

    idx_list = df.index[df["datetime"] == entry_time].tolist()
    if not idx_list:
        return 0

    idx = idx_list[0]

    if trade_type == "BUY":
        target = entry + (entry - sl) * rr
    else:
        target = entry - (sl - entry) * rr

    for i in range(idx + 1, len(df)):
        candle = df.iloc[i]

        if trade_type == "BUY":
            if candle["low"] <= sl:
                return 0
            if candle["high"] >= target:
                return 1
        else:
            if candle["high"] >= sl:
                return 0
            if candle["low"] <= target:
                return 1

    return 0  # BE treated as LOSS

# =========================
# INIT STRATEGIES
# =========================
strategies = [
    VWAPPullbackStrategy(rr=RR),
    PivotStrategy(rr=RR),
    ABCDStrategy(rr=RR),
]

all_rows = []

# =========================
# BUILD TRAINING DATA
# =========================
for strat in strategies:
    trades = strat.generate_trades(df)

    for trade in trades:
        result = evaluate_trade(df, trade)

        all_rows.append({
            "strategy": trade["strategy"],
            "entry": trade["entry"],
            "stoploss": trade["stoploss"],
            "rr": trade["rr"],
            "vwap_distance": trade.get("vwap_distance", 0),
            "candle_size": abs(trade["entry"] - trade["stoploss"]),
            "atr": trade.get("atr", 0),
            "pattern_strength": trade.get("pattern_strength", 0),
            "result": result
        })

# =========================
# SAVE CSV
# =========================
df_out = pd.DataFrame(all_rows)
df_out.to_csv(OUTPUT_PATH, index=False)

print("✅ Training data created successfully")
print(df_out["strategy"].value_counts())
print("Total samples:", len(df_out))
