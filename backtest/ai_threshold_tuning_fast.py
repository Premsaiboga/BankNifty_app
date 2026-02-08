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
# INDICATORS
# =========================
df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
df["vwap"] = (
    df.groupby("date")["tp"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

hl = df["high"] - df["low"]
hc = (df["high"] - df["close"].shift()).abs()
lc = (df["low"] - df["close"].shift()).abs()
df["tr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(14).mean()

daily = df.groupby("date").agg(
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last")
)

daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3
daily["r1"] = 2 * daily["pivot"] - daily["low"]
daily["s1"] = 2 * daily["pivot"] - daily["high"]

df = df.merge(daily[["pivot", "r1", "s1"]], left_on="date", right_index=True, how="left")

# =========================
# STRATEGIES
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
# EVALUATE ALL TRADES ONCE
# =========================
rows = []

for i, trade in enumerate(raw_trades, 1):

    if i % 1000 == 0:
        print(f"Processed {i}/{len(raw_trades)} trades...")

    idx_list = df.index[df["datetime"] == trade["time"]].tolist()
    if not idx_list:
        continue

    idx = idx_list[0]
    entry = trade["entry"]
    sl = trade["stoploss"]
    rr = trade["rr"]
    ttype = trade["type"]

    target = (
        entry + (entry - sl) * rr
        if ttype == "BUY"
        else entry - (sl - entry) * rr
    )

    result = "LOSS"
    for i in range(idx + 1, len(df)):
        c = df.iloc[i]
        if ttype == "BUY":
            if c["low"] <= sl:
                break
            if c["high"] >= target:
                result = "WIN"
                break
        else:
            if c["high"] >= sl:
                break
            if c["low"] <= target:
                result = "WIN"
                break

    X = pd.DataFrame([{
        "strategy": {"VWAP_PULLBACK": 0, "PIVOT": 1, "ABCD": 2}[trade["strategy"]],
        "vwap_distance": trade.get("vwap_distance", 0),
        "candle_size": abs(entry - sl),
        "atr": trade.get("atr", 0),
        "pattern_strength": trade.get("pattern_strength", 0),
        "rr": rr
    }])

    prob = model.predict_proba(
        scaler.transform(X[features])
    )[0][1]

    rows.append({
        "time": trade["time"],
        "strategy": trade["strategy"],
        "prob": prob,
        "result": result
    })

df_all = pd.DataFrame(rows)
df_all["month"] = pd.to_datetime(df_all["time"]).dt.to_period("M")

# =========================
# THRESHOLD ANALYSIS (FAST)
# =========================
summary = []

for t in THRESHOLDS:
    sub = df_all[df_all["prob"] >= t]

    if sub.empty:
        continue

    wins = (sub["result"] == "WIN").sum()
    losses = (sub["result"] == "LOSS").sum()

    summary.append({
        "threshold": t,
        "trades": len(sub),
        "win_rate": round(wins / len(sub), 3),
        "net_R": wins * RR - losses,
        "avg_trades_per_month": round(len(sub) / sub["month"].nunique(), 1)
    })

print("\n📊 AI THRESHOLD TUNING RESULTS\n")
print(pd.DataFrame(summary).sort_values("net_R", ascending=False))
