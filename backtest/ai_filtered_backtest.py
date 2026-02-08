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
# PROB_THRESHOLD = 0.6
# MAX_LOOKAHEAD = 60   # 🔥 critical speed fix

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df.reset_index(drop=True, inplace=True)

# =========================
# LOAD MODEL
# =========================
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

# =========================
# INDICATORS (SAME AS TRAINING)
# =========================
df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
df["vwap"] = (
    df.groupby(df["datetime"].dt.date)["tp"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

hl = df["high"] - df["low"]
hc = (df["high"] - df["close"].shift()).abs()
lc = (df["low"] - df["close"].shift()).abs()
df["tr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()

df["date"] = df["datetime"].dt.date

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
# FAST TRADE EVALUATOR
# =========================
def evaluate_trade(df, trade):
    idx_list = df.index[df["datetime"] == trade["time"]].tolist()
    if not idx_list:
        return "LOSS"

    idx = idx_list[0]
    entry = trade["entry"]
    sl = trade["stoploss"]
    rr = trade["rr"]
    ttype = trade["type"]

    if ttype == "BUY":
        target = entry + (entry - sl) * rr
    else:
        target = entry - (sl - entry) * rr

    end = min(idx + MAX_LOOKAHEAD, len(df))

    for i in range(idx + 1, end):
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
strategies = [
    VWAPPullbackStrategy(rr=RR),
    PivotStrategy(rr=RR),
    ABCDStrategy(rr=RR),
]

raw_trades = []
for s in strategies:
    raw_trades.extend(s.generate_trades(df))

print("Total raw trades:", len(raw_trades))

# =========================
# AI FILTER + BACKTEST
# =========================
rows = []

for i, trade in enumerate(raw_trades):
    if i % 1000 == 0:
        print(f"Processed {i}/{len(raw_trades)} trades...")

    X = pd.DataFrame([{
        "strategy": {"VWAP_PULLBACK": 0, "PIVOT": 1, "ABCD": 2}[trade["strategy"]],
        "vwap_distance": trade.get("vwap_distance", 0),
        "candle_size": abs(trade["entry"] - trade["stoploss"]),
        "atr": trade.get("atr", 0),
        "pattern_strength": trade.get("pattern_strength", 0),
        "rr": trade["rr"]
    }])

    X_scaled = scaler.transform(X[features])
    prob = model.predict_proba(X_scaled)[0][1]

    if prob < PROB_THRESHOLD:
        continue

    result = evaluate_trade(df, trade)

    rows.append({
        "time": trade["time"],
        "strategy": trade["strategy"],
        "prob": prob,
        "result": result
    })

# =========================
# RESULTS
# =========================
res = pd.DataFrame(rows)

if res.empty:
    print("❌ No AI-approved trades")
    exit()

res["month"] = pd.to_datetime(res["time"]).dt.to_period("M")

monthly = res.groupby("month").agg(
    trades=("result", "count"),
    wins=("result", lambda x: (x == "WIN").sum()),
    losses=("result", lambda x: (x == "LOSS").sum())
)

monthly["net_R"] = monthly["wins"] * RR - monthly["losses"]

print("\n📊 AI FILTERED MONTHLY RESULTS\n")
print(monthly)

print("\n📈 OVERALL STATS")
print("AI Trades :", len(res))
print("Win rate  :", (res["result"] == "WIN").mean())
print("Net R     :", monthly["net_R"].sum())

print("\n📌 Strategy-wise AI trades")
print(res["strategy"].value_counts())
