import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "rr",
    "vwap_distance",
    "candle_size",
    "atr",
    "sl_points",
    "entry_vs_vwap"
]

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # ===== LABEL =====
    df["label"] = df["result"].apply(lambda x: 1 if x.upper() == "WIN" else 0)

    # ===== FEATURE ENGINEERING =====
    df["sl_points"] = (df["entry"] - df["stoploss"]).abs()
    df["entry_vs_vwap"] = df["vwap_distance"] / (df["atr"] + 1e-6)

    # ===== SELECT FEATURES =====
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # ===== SCALE =====
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
