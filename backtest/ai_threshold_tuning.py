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
# INDICATORS (same as training)
# =========================
df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
df["vwap"] = (
    df.groupby(df["date"])["tp"]
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

print("\nTotal raw trades:", len(raw_trades))

# =========================
# TRADE EVALUATOR
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
# THRESHOLD LOOP
# =========================
summary_rows = []

for threshold in THRESHOLDS:
    rows = []

    for trade in raw_trades:
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

        if prob < threshold:
            continue

        result = evaluate_trade(df, trade)
        rows.append((trade["time"], result))

    if not rows:
        continue

    res = pd.DataFrame(rows, columns=["time", "result"])
    res["month"] = pd.to_datetime(res["time"]).dt.to_period("M")

    trades = len(res)
    wins = (res["result"] == "WIN").sum()
    losses = (res["result"] == "LOSS").sum()
    net_r = wins * RR - losses
    winrate = wins / trades

    summary_rows.append({
        "threshold": threshold,
        "trades": trades,
        "win_rate": round(winrate, 3),
        "net_R": net_r,
        "avg_trades_per_month": round(trades / res["month"].nunique(), 1)
    })

# =========================
# FINAL OUTPUT
# =========================
summary_df = pd.DataFrame(summary_rows)

print("\n📊 AI THRESHOLD TUNING RESULTS\n")
print(summary_df.sort_values("net_R", ascending=False))
