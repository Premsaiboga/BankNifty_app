import sys
from pathlib import Path
import pandas as pd
import joblib

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
MODEL_PATH = PROJECT_ROOT / "ml/model.pkl"

RR = 4
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df["date"] = df["datetime"].dt.date

# =========================
# LOAD MODEL
# =========================
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

# =========================
# INDICATORS (TRAINING SAME)
# =========================
df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
df["vwap"] = (
    df.groupby("date")["tp"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift()).abs()
low_close = (df["low"] - df["close"].shift()).abs()
df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()

daily = df.groupby("date").agg(
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last")
)

daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3
daily["r1"] = 2 * daily["pivot"] - daily["low"]
daily["s1"] = 2 * daily["pivot"] - daily["high"]

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
    entry = trade["entry"]
    sl = trade["stoploss"]
    rr = trade["rr"]
    ttype = trade["type"]
    time = trade["time"]

    idxs = df.index[df["datetime"] == time].tolist()
    if not idxs:
        return "LOSS"

    idx = idxs[0]
    target = (
        entry + (entry - sl) * rr
        if ttype == "BUY"
        else entry - (sl - entry) * rr
    )

    for i in range(idx + 1, len(df)):
        c = df.iloc[i]
        if ttype == "BUY":
            if c["low"] <= sl:
                return "LOSS"
            if c["high"] >= target:
                return "WIN"
        else:
            if c["high"] >= sl:
                return "LOSS"
            if c["low"] <= target:
                return "WIN"

    return "LOSS"

# =========================
# INIT STRATEGIES
# =========================
strategy_objs = {
    "VWAP_PULLBACK": VWAPPullbackStrategy(rr=RR),
    "PIVOT": PivotStrategy(rr=RR),
    "ABCD": ABCDStrategy(rr=RR),
}

# =========================
# MAIN LOOP
# =========================
all_results = []

for strat_name, strat in strategy_objs.items():

    trades = strat.generate_trades(df)
    print(f"\n🔹 {strat_name} raw trades: {len(trades)}")

    for threshold in THRESHOLDS:

        print(f"  ▶ Testing threshold {threshold}")

        for i, trade in enumerate(trades, 1):

            if i % 1000 == 0:
                print(f"    {strat_name} | {threshold} | {i}/{len(trades)}")

            X = pd.DataFrame([{
                "strategy": {"VWAP_PULLBACK": 0, "PIVOT": 1, "ABCD": 2}[trade["strategy"]],
                "vwap_distance": trade.get("vwap_distance", 0),
                "candle_size": abs(trade["entry"] - trade["stoploss"]),
                "atr": trade.get("atr", 0),
                "pattern_strength": trade.get("pattern_strength", 0),
                "rr": trade["rr"]
            }])

            prob = model.predict_proba(
                scaler.transform(X[features])
            )[0][1]

            if prob < threshold:
                continue

            result = evaluate_trade(df, trade)

            all_results.append({
                "time": trade["time"],
                "strategy": strat_name,
                "threshold": threshold,
                "result": result
            })

# =========================
# RESULTS
# =========================
res = pd.DataFrame(all_results)
res["month"] = pd.to_datetime(res["time"]).dt.to_period("M")

monthly = res.groupby(["strategy", "threshold", "month"]).agg(
    trades=("result", "count"),
    wins=("result", lambda x: (x == "WIN").sum()),
    losses=("result", lambda x: (x == "LOSS").sum())
)

monthly["net_R"] = monthly["wins"] * RR - monthly["losses"]

overall = res.groupby(["strategy", "threshold"]).agg(
    trades=("result", "count"),
    win_rate=("result", lambda x: (x == "WIN").mean())
).reset_index()

overall["net_R"] = (
    res.assign(r=lambda x: x["result"].map({"WIN": RR, "LOSS": -1}))
    .groupby(["strategy", "threshold"])["r"]
    .sum()
    .values
)

print("\n📊 MONTHLY RESULTS (strategy × threshold)\n")
print(monthly)

print("\n📈 OVERALL RESULTS\n")
print(overall)
